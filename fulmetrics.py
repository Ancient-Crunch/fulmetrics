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
        
        # Remove SKUs that are purely numeric and underscores
        # This regex pattern matches strings that contain only digits and underscores
        numeric_underscore_pattern = r'^[\d_]+$'
        df = df[~df['item_sku'].str.match(numeric_underscore_pattern, na=False)]
        
        filtered_count = original_count - len(df)
        
        print(f"Successfully loaded {len(df)} rows from {filepath}")
        if filtered_count > 0:
            print(f"Filtered out {filtered_count} rows with missing SKU, zero/missing quantity, or numeric-only SKUs")
        print(f"Date range: {df['order_date'].min()} to {df['order_date'].max()}")
        print(f"Number of unique SKUs: {df['item_sku'].nunique()}")
        print(f"Number of unique orders: {df['order_number'].nunique()}")
        
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
    
    SKU formats:
    - Single bags: VANDOR5, MASAOG5 (no hyphens)
    - Simple multi-packs: VANDOR5-6X (6 bags of VANDOR5)
    - Mix packs: MASAMIX5-OG2X-LI3X-CO1X (mix of different masa flavors)
    - Multi-brand boxes: MASAMIX5-OG10X-VANDMIX5-OR10X
    
    Args:
        df (pd.DataFrame): The order data with SKUs
        
    Returns:
        pd.DataFrame: Expanded data with one row per individual item type
    """
    expanded_rows = []
    unparseable_skus = []
    
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
            components = sku.split('-')
            
            # Check if it's a simple multi-pack (e.g., "VANDOR5-6X")
            if len(components) == 2 and components[1].endswith('X'):
                try:
                    single_sku = components[0]
                    pack_size_str = components[1][:-1]  # Remove 'X'
                    pack_size = int(pack_size_str)
                    
                    expanded_rows.append({
                        'order_number': row['order_number'],
                        'order_date': row['order_date'],
                        'single_bag_sku': single_sku,
                        'quantity': base_qty * pack_size,
                        'original_sku': sku,
                        'store_name': row['store_name'],
                        'customer_email': row['customer_email'],
                        'order_total': row['order_total']
                    })
                except ValueError:
                    # If we can't parse the pack size, treat as unparseable
                    unparseable_skus.append(sku)
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
                # Complex SKU - could be mix pack or multi-brand box
                parsed_items = []
                i = 0
                
                while i < len(components):
                    component = components[i].upper()
                    
                    # Check if this is a MIX SKU (MASAMIX5 or VANDMIX5)
                    if 'MIX' in component:
                        # Extract the brand and size from the mix component
                        if component.startswith('MASAMIX'):
                            brand = 'MASA'
                            size = component.replace('MASAMIX', '')  # Get the size (e.g., '5')
                        elif component.startswith('VANDMIX'):
                            brand = 'VAND'
                            size = component.replace('VANDMIX', '')
                        else:
                            # Unknown mix format, skip
                            i += 1
                            continue
                        
                        # Process flavor-quantity pairs that follow
                        i += 1
                        while i < len(components) and components[i].endswith('X'):
                            try:
                                # Extract flavor code and quantity
                                flavor_qty = components[i][:-1]  # Remove 'X'
                                
                                # Find where the number starts
                                for j, char in enumerate(flavor_qty):
                                    if char.isdigit():
                                        flavor_code = flavor_qty[:j]
                                        qty_str = flavor_qty[j:]
                                        break
                                else:
                                    # No digit found, skip
                                    i += 1
                                    continue
                                
                                qty = int(qty_str)
                                
                                # Construct the single bag SKU
                                single_sku = f"{brand}{flavor_code}{size}"
                                
                                parsed_items.append({
                                    'single_sku': single_sku,
                                    'quantity': qty
                                })
                                
                                i += 1
                            except (ValueError, IndexError):
                                i += 1
                                continue
                    else:
                        # Not a MIX component, check if next component is a quantity
                        if i + 1 < len(components) and components[i + 1].endswith('X'):
                            try:
                                single_sku = component
                                pack_size = int(components[i + 1][:-1])
                                
                                parsed_items.append({
                                    'single_sku': single_sku,
                                    'quantity': pack_size
                                })
                                i += 2
                            except ValueError:
                                i += 1
                        else:
                            i += 1
                
                # Create rows for each parsed item
                if parsed_items:
                    # Calculate revenue split
                    total_items = len(parsed_items)
                    revenue_per_item = row['order_total'] / total_items if total_items > 0 else row['order_total']
                    
                    for item in parsed_items:
                        expanded_rows.append({
                            'order_number': row['order_number'],
                            'order_date': row['order_date'],
                            'single_bag_sku': item['single_sku'],
                            'quantity': base_qty * item['quantity'],
                            'original_sku': sku,
                            'store_name': row['store_name'],
                            'customer_email': row['customer_email'],
                            'order_total': revenue_per_item
                        })
                else:
                    # Couldn't parse anything, keep original
                    unparseable_skus.append(sku)
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
    
    # Create DataFrame from expanded rows
    expanded_df = pd.DataFrame(expanded_rows)
    
    print(f"\nExpanded {len(df)} original rows to {len(expanded_df)} individual item rows")
    print(f"Number of unique single bag SKUs: {expanded_df['single_bag_sku'].nunique()}")
    
    if unparseable_skus:
        unique_unparseable = list(set(unparseable_skus))
        print(f"\nWarning: Could not parse {len(unique_unparseable)} unique SKUs (treated as single items):")
        for sku in unique_unparseable[:10]:  # Show first 10
            print(f"  - {sku}")
        if len(unique_unparseable) > 10:
            print(f"  ... and {len(unique_unparseable) - 10} more")
    
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
    
    # Parse SKUs to get individual bag items
    expanded_df = parse_sku_to_items(df)
    
    # Calculate daily individual bag summary
    print("\n" + "="*50)
    print("DAILY INDIVIDUAL BAG QUANTITIES")
    print("="*50)
    daily_bag_summary = get_daily_single_bag_summary(expanded_df)
    
    # Show top bags for each day
    for date in daily_bag_summary['date'].unique():
        day_data = daily_bag_summary[daily_bag_summary['date'] == date]
        print(f"\n{date}:")
        print(day_data.head(10).to_string(index=False))
    
    # Calculate overall individual bag summary
    print("\n" + "="*50)
    print("OVERALL INDIVIDUAL BAG SUMMARY")
    print("="*50)
    bag_summary = get_single_bag_summary(expanded_df)
    print(bag_summary.head(20).to_string(index=False))
    
    # Calculate daily SKU summary (original SKUs as ordered)
    print("\n" + "="*50)
    print("DAILY SKU QUANTITIES (AS ORDERED)")
    print("="*50)
    daily_summary = get_daily_sku_summary(df)
    
    # Show top SKUs for each day
    for date in daily_summary['date'].unique():
        day_data = daily_summary[daily_summary['date'] == date]
        print(f"\n{date}:")
        print(day_data.head(10).to_string(index=False))
    
    # Calculate overall SKU summary
    print("\n" + "="*50)
    print("OVERALL SKU SUMMARY (AS ORDERED)")
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
    
    # Save individual bag summaries
    daily_bag_summary.to_csv(f'{base_name}_daily_bag_summary.csv', index=False)
    print(f"Saved daily individual bag summary to: {base_name}_daily_bag_summary.csv")
    
    bag_summary.to_csv(f'{base_name}_overall_bag_summary.csv', index=False)
    print(f"Saved overall individual bag summary to: {base_name}_overall_bag_summary.csv")
    
    # Save original SKU summaries
    daily_summary.to_csv(f'{base_name}_daily_sku_summary.csv', index=False)
    print(f"Saved daily SKU summary to: {base_name}_daily_sku_summary.csv")
    
    sku_summary.to_csv(f'{base_name}_overall_sku_summary.csv', index=False)
    print(f"Saved overall SKU summary to: {base_name}_overall_sku_summary.csv")
    
    # Save the expanded dataset for reference
    expanded_df.to_csv(f'{base_name}_expanded_items.csv', index=False)
    print(f"Saved expanded items dataset to: {base_name}_expanded_items.csv")
    
    return df, expanded_df

if __name__ == "__main__":
    main()