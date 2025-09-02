#!/usr/bin/env python3
"""
ShipStation Daily Metrics Script
Analyzes daily order exports to understand SKU quantities and order metrics
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

def load_shipstation_data(filepath):
    """
    Load ShipStation CSV export into a pandas DataFrame
    
    Args:
        filepath (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded data with proper column names
    """
    try:
        # Read the CSV file
        df = pd.read_csv(filepath)
        
        # Clean up column names (remove spaces and special characters for easier access)
        df.columns = [
            'order_number',
            'customer_email', 
            'order_date',
            'item_sku',
            'item_qty',
            'order_total',
            'store_name'
        ]
        
        # Convert date column to datetime
        df['order_date'] = pd.to_datetime(df['order_date'])
        
        # Ensure numeric columns are properly typed
        df['item_qty'] = pd.to_numeric(df['item_qty'], errors='coerce')
        df['order_total'] = pd.to_numeric(df['order_total'], errors='coerce')

        # Store original row count
        original_count = len(df)
        
        # Remove rows with missing SKU or zero/missing quantity
        df = df[df['item_sku'].notna() & (df['item_sku'] != '')]  # Remove missing/empty SKUs
        df = df[df['item_qty'].notna() & (df['item_qty'] > 0)]    # Remove missing or zero quantities
        
        filtered_count = original_count - len(df)

        print(f"Successfully loaded {len(df)} rows from {filepath}")
        if filtered_count > 0:
            print(f"Filtered out {filtered_count} rows with missing SKU or zero/missing quantity")
        print(f"Date range: {df['order_date'].min()} to {df['order_date'].max()}")
        print(f"Number of unique SKUs: {df['item_sku'].nunique()}")
        print(f"Number of unique orderlines: {df['order_number'].nunique()}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found!")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)


def parse_sku_to_items(df):
    """
    Parse SKUs to break down multi-bag boxes into individual items
    
    Args:
        df (pd.DataFrame): The order data with SKUs
        
    Returns:
        pd.DataFrame: Expanded data with one row per individual item type
    """
    expanded_rows = []
    
    for _, row in df.iterrows():
        sku = row['item_sku']
        base_qty = row['item_qty']
        
        # Check if it's a single bag SKU (no hyphens)
        if '-' not in sku:
            # Single bag SKU - just copy the row as is
            expanded_rows.append({
                'order_number': row['order_number'],
                'order_date': row['order_date'],
                'single_bag_sku': sku,
                'quantity': base_qty,
                'original_sku': sku,
                'store_name': row['store_name'],
                'customer_email': row['customer_email'],
                'order_total': row['order_total']
            })
        else:
            # Multi-bag box SKU - need to parse it
            # Split by hyphens to get components
            components = sku.split('-')
            
            # Check if it's a simple multi-pack (e.g., "VANDOR5-6X")
            if len(components) == 2 and components[1].endswith('X'):
                single_sku = components[0]
                pack_size = int(components[1][:-1])  # Remove 'X' and convert to int
                
                expanded_rows.append({
                    'order_number': row['order_number'],
                    'order_date': row['order_date'],
                    'single_bag_sku': single_sku,
                    'quantity': base_qty * pack_size,  # Total individual bags
                    'original_sku': sku,
                    'store_name': row['store_name'],
                    'customer_email': row['customer_email'],
                    'order_total': row['order_total']
                })
            else:
                # Multi-flavor bundle (e.g., "MASAMIX5-OR10X-VANDMIX5-OR10X")
                # Process each flavor-quantity pair
                i = 0
                while i < len(components):
                    if i + 1 < len(components) and components[i + 1].endswith('X'):
                        single_sku = components[i]
                        pack_size = int(components[i + 1][:-1])
                        
                        expanded_rows.append({
                            'order_number': row['order_number'],
                            'order_date': row['order_date'],
                            'single_bag_sku': single_sku,
                            'quantity': base_qty * pack_size,  # Total individual bags
                            'original_sku': sku,
                            'store_name': row['store_name'],
                            'customer_email': row['customer_email'],
                            'order_total': row['order_total'] / len([c for j, c in enumerate(components) if j % 2 == 0 and j + 1 < len(components)])  # Split revenue proportionally
                        })
                        i += 2
                    else:
                        # Skip if we can't parse properly
                        print(f"Warning: Could not parse SKU component: {components[i]} in SKU: {sku}")
                        i += 1
    
    # Create DataFrame from expanded rows
    expanded_df = pd.DataFrame(expanded_rows)
    
    print(f"\nExpanded {len(df)} original rows to {len(expanded_df)} individual item rows")
    print(f"Number of unique single bag SKUs: {expanded_df['single_bag_sku'].nunique()}")
    
    return expanded_df


def get_single_bag_summary(expanded_df):
    """
    Get summary of individual bag quantities across all orders
    
    Args:
        expanded_df (pd.DataFrame): The expanded data with individual items
        
    Returns:
        pd.DataFrame: Summary of total quantities by single bag SKU
    """
    bag_summary = expanded_df.groupby('single_bag_sku').agg({
        'quantity': 'sum',
        'order_number': 'nunique',
        'order_total': 'sum'
    }).reset_index()
    
    bag_summary.columns = ['single_bag_sku', 'total_bags', 'unique_orders', 'total_revenue']
    bag_summary = bag_summary.sort_values('total_bags', ascending=False)
    
    return bag_summary


def get_daily_single_bag_summary(expanded_df):
    """
    Calculate daily individual bag quantities
    
    Args:
        expanded_df (pd.DataFrame): The expanded data with individual items
        
    Returns:
        pd.DataFrame: Summary of bag quantities by date
    """
    # Group by date and single bag SKU, sum the quantities
    daily_bags = expanded_df.groupby([expanded_df['order_date'].dt.date, 'single_bag_sku'])['quantity'].sum().reset_index()
    daily_bags.columns = ['date', 'single_bag_sku', 'total_bags']
    
    # Sort by date and quantity (highest first)
    daily_bags = daily_bags.sort_values(['date', 'total_bags'], ascending=[True, False])
    
    return daily_bags


def get_daily_sku_summary(df):
    """
    Calculate daily SKU quantities owed
    
    Args:
        df (pd.DataFrame): The order data
        
    Returns:
        pd.DataFrame: Summary of SKU quantities by date
    """
    # Group by date and SKU, sum the quantities
    daily_sku = df.groupby([df['order_date'].dt.date, 'item_sku'])['item_qty'].sum().reset_index()
    daily_sku.columns = ['date', 'sku', 'total_qty']
    
    # Sort by date and quantity (highest first)
    daily_sku = daily_sku.sort_values(['date', 'total_qty'], ascending=[True, False])
    
    return daily_sku

def get_sku_summary(df):
    """
    Get overall SKU summary across all dates
    
    Args:
        df (pd.DataFrame): The order data
        
    Returns:
        pd.DataFrame: Summary of total quantities by SKU
    """
    sku_summary = df.groupby('item_sku').agg({
        'item_qty': 'sum',
        'order_number': 'nunique',
        'order_total': 'sum'
    }).reset_index()
    
    sku_summary.columns = ['sku', 'total_qty', 'unique_orders', 'total_revenue']
    sku_summary = sku_summary.sort_values('total_qty', ascending=False)
    
    return sku_summary

def main():
    """
    Main function to run the analysis
    """
    # Check if filename was provided
    if len(sys.argv) < 2:
        print("Usage: python shipstation_metrics.py <csv_filename>")
        sys.exit(1)
    
    filepath = sys.argv[1]
    
    # Load the data
    df = load_shipstation_data(filepath)
    
    # Calculate daily SKU summary
    print("\n" + "="*50)
    print("DAILY SKU QUANTITIES")
    print("="*50)
    daily_summary = get_daily_sku_summary(df)
    
    # Show top SKUs for each day
    for date in daily_summary['date'].unique():
        day_data = daily_summary[daily_summary['date'] == date]
        print(f"\n{date}:")
        print(day_data.head(10).to_string(index=False))
    
    # Calculate overall SKU summary
    print("\n" + "="*50)
    print("OVERALL SKU SUMMARY")
    print("="*50)
    sku_summary = get_sku_summary(df)
    print(sku_summary.head(20).to_string(index=False))
    
    # Store summary
    print("\n" + "="*50)
    print("STORE BREAKDOWN")
    print("="*50)
    store_summary = df.groupby('store_name').agg({
        'order_number': 'nunique',
        'item_qty': 'sum',
        'order_total': 'sum'
    }).reset_index()
    store_summary.columns = ['store', 'unique_orders', 'total_items', 'total_revenue']
    print(store_summary.to_string(index=False))
    
    # Save summaries to CSV files (optional)
    print("\n" + "="*50)
    print("SAVING SUMMARIES")
    print("="*50)
    
    # Create output filename based on input
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    
    daily_summary.to_csv(f'{base_name}_daily_sku_summary.csv', index=False)
    print(f"Saved daily SKU summary to: {base_name}_daily_sku_summary.csv")
    
    sku_summary.to_csv(f'{base_name}_overall_sku_summary.csv', index=False)
    print(f"Saved overall SKU summary to: {base_name}_overall_sku_summary.csv")
    
    return df

if __name__ == "__main__":
    main()