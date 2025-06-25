import pandas as pd

def prepare_weekly_sales(sales_df):
    sales_df['Date'] = pd.to_datetime(sales_df['Date'])
    sales_df['Week'] = sales_df['Date'].dt.to_period('W').apply(lambda r: r.start_time)
    weekly_sales = sales_df.groupby(['SKU', 'Zone', 'Week'])['Quantity_Sold'].sum().reset_index()
    return weekly_sales

def merge_trends(weekly_sales, trends_df):
    trends_df['Date'] = pd.to_datetime(trends_df['Date'])
    trends_df['Week'] = trends_df['Date'].dt.to_period('W').apply(lambda r: r.start_time)
    weekly_trends = trends_df.groupby(['SKU', 'Zone', 'Week'])['Trend_Score'].mean().reset_index()
    merged = pd.merge(weekly_sales, weekly_trends, on=['SKU', 'Zone', 'Week'], how='left')
    return merged

def merge_holidays(merged_df, holidays_df):
    holidays_df['Date'] = pd.to_datetime(holidays_df['Date'])
    holidays_df['Week'] = holidays_df['Date'].dt.to_period('W').apply(lambda r: r.start_time)
    holidays_df['Is_Holiday'] = 1
    holiday_flags = holidays_df.groupby(['Zone', 'Week'])['Is_Holiday'].max().reset_index()
    final_df = pd.merge(merged_df, holiday_flags, on=['Zone', 'Week'], how='left')
    final_df['Is_Holiday'] = final_df['Is_Holiday'].fillna(0).astype(int)
    return final_df

def generate_features(data_dict):
    sales_df = data_dict['sales']
    trends_df = data_dict['trends']
    holidays_df = data_dict['holidays']

    weekly_sales = prepare_weekly_sales(sales_df)
    merged_with_trends = merge_trends(weekly_sales, trends_df)
    full_features = merge_holidays(merged_with_trends, holidays_df)

    return full_features

