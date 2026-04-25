# %% [markdown]
# ## Interactive Business Dashboard in Streamlit

# %%
import pandas as pd
import numpy as np
from datetime import datetime

# %%
class DataPreprocessor:
    def __init__(self, file_path):
        """
        Initialize the preprocessor with the CSV file path
        
        Parameters:
        file_path (str): Path to the Global Superstore CSV file
        """
        self.file_path = file_path
        self.df = None
        
    def load_data(self):
        """
        Load the CSV data and perform initial setup
        """
        try:
            self.df = pd.read_csv(self.file_path, encoding='latin1')
            print(f"Data loaded successfully! Shape: {self.df.shape}")
            return self.df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def explore_initial_stats(self):
        """
        Display initial dataset statistics
        """
        if self.df is not None:
            print("\n" + "="*50)
            print("INITIAL DATA EXPLORATION")
            print("="*50)
            print(f"Total Rows: {self.df.shape[0]}")
            print(f"Total Columns: {self.df.shape[1]}")
            print(f"\nColumn Names: {list(self.df.columns)}")
            print(f"\nMissing Values:")
            print(self.df.isnull().sum())
            print(f"\nData Types:")
            print(self.df.dtypes)
    
    def clean_data(self):
        """
        Perform all data cleaning operations
        """
        if self.df is None:
            return None
            
        print("\n" + "="*50)
        print("STARTING DATA CLEANING PROCESS")
        print("="*50)
        
        # 1. Remove duplicates
        initial_rows = len(self.df)
        self.df = self.df.drop_duplicates()
        print(f"Removed {initial_rows - len(self.df)} duplicate rows")
        
        # 2. Handle missing values
        missing_before = self.df.isnull().sum().sum()
        
        # Drop Postal Code (too many missing values, not critical for dashboard)
        if 'Postal Code' in self.df.columns:
            self.df = self.df.drop('Postal Code', axis=1)
            print("Dropped 'Postal Code' column (high missing values, not needed)")
        
        # Fill remaining missing values appropriately
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                self.df[col] = self.df[col].fillna('Unknown')
            else:
                self.df[col] = self.df[col].fillna(0)
        
        missing_after = self.df.isnull().sum().sum()
        print(f"Handled missing values: Reduced from {missing_before} to {missing_after}")
        
        # 3. Convert date columns to datetime
        date_columns = ['Order Date', 'Ship Date']
        for col in date_columns:
            if col in self.df.columns:
                try:
                    self.df[col] = pd.to_datetime(self.df[col], format='%m/%d/%Y', errors='coerce')
                    print(f"Converted {col} to datetime format")
                except:
                    try:
                        self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                        print(f"Converted {col} to datetime format")
                    except:
                        print(f"Could not convert {col} to datetime")
        
        # 4. Ensure numeric columns are proper type
        numeric_columns = ['Sales', 'Profit', 'Quantity', 'Discount']
        for col in numeric_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0)
                print(f"Ensured {col} is numeric")
        
        # 5. Remove negative profits for certain analyses (optional - keep for dashboard)
        # We'll keep negative profits for accurate reporting
        
        # 6. Clean text columns (strip whitespace, standardize)
        text_columns = ['Segment', 'Region', 'Category', 'Sub-Category', 'Customer Name']
        for col in text_columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].str.strip().str.title()
                print(f"Cleaned text in {col}")
        
        # 7. Add calculated columns for better analysis
        # Profit Margin
        self.df['Profit_Margin'] = np.where(
            self.df['Sales'] != 0, 
            (self.df['Profit'] / self.df['Sales']) * 100, 
            0
        )
        print("Added 'Profit_Margin' column")
        
        # Year and Month for time analysis
        if 'Order Date' in self.df.columns:
            self.df['Year'] = self.df['Order Date'].dt.year
            self.df['Month'] = self.df['Order Date'].dt.month
            self.df['Month_Name'] = self.df['Order Date'].dt.strftime('%B')
            print("Added time-based columns (Year, Month, Month_Name)")
        
        print("\n" + "="*50)
        print("DATA CLEANING COMPLETED")
        print("="*50)
        print(f"Final Data Shape: {self.df.shape}")
        
        return self.df
    
    def validate_data(self):
        """
        Validate data quality after cleaning
        """
        if self.df is None:
            return False
            
        print("\n" + "="*50)
        print("DATA VALIDATION")
        print("="*50)
        
        # Check for any remaining nulls
        remaining_nulls = self.df.isnull().sum().sum()
        print(f"Remaining null values: {remaining_nulls}")
        
        # Check numeric ranges
        if 'Sales' in self.df.columns:
            print(f"\nSales range: ${self.df['Sales'].min():.2f} to ${self.df['Sales'].max():.2f}")
        if 'Profit' in self.df.columns:
            print(f"Profit range: ${self.df['Profit'].min():.2f} to ${self.df['Profit'].max():.2f}")
        
        # Check unique values in key columns
        print(f"\nUnique Segments: {self.df['Segment'].nunique()}")
        print(f"Unique Regions: {self.df['Region'].nunique()}")
        print(f"Unique Categories: {self.df['Category'].nunique()}")
        print(f"Unique Customers: {self.df['Customer Name'].nunique()}")
        
        return True
    
    def get_summary_stats(self):
        """
        Generate summary statistics for the dashboard
        """
        if self.df is None:
            return {}
        
        summary = {
            'total_sales': self.df['Sales'].sum(),
            'total_profit': self.df['Profit'].sum(),
            'avg_profit_margin': self.df['Profit_Margin'].mean(),
            'total_orders': self.df.shape[0],
            'total_customers': self.df['Customer Name'].nunique(),
            'unique_products': self.df['Sub-Category'].nunique()
        }
        
        return summary
    
    def save_cleaned_data(self, output_path='cleaned_superstore.csv'):
        """
        Save the cleaned dataset to CSV
        """
        if self.df is not None:
            self.df.to_csv(output_path, index=False)
            print(f"\nCleaned data saved to '{output_path}'")
            return True
        return False

# %%
# Additional helper functions for specific analyses
def get_top_customers(df, n=5):
    """
    Get top N customers by sales
    
    Parameters:
    df: Cleaned DataFrame
    n: Number of top customers to return
    
    Returns:
    DataFrame with top customers and their metrics
    """
    top_customers = df.groupby('Customer Name').agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Quantity': 'sum',
        'Order ID': 'count'
    }).round(2).reset_index()
    
    top_customers.columns = ['Customer Name', 'Total Sales', 'Total Profit', 'Total Quantity', 'Order Count']
    top_customers = top_customers.sort_values('Total Sales', ascending=False).head(n)
    
    return top_customers

# %%
def get_segment_performance(df):
    """
    Get performance metrics by customer segment
    """
    segment_stats = df.groupby('Segment').agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Quantity': 'sum',
        'Order ID': 'count'
    }).round(2).reset_index()
    
    segment_stats.columns = ['Segment', 'Total Sales', 'Total Profit', 'Total Quantity', 'Order Count']
    segment_stats['Profit Margin (%)'] = (segment_stats['Total Profit'] / segment_stats['Total Sales'] * 100).round(2)
    
    return segment_stats

# %%
def get_region_performance(df):
    """
    Get performance metrics by region
    """
    region_stats = df.groupby('Region').agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Quantity': 'sum'
    }).round(2).reset_index()
    
    region_stats['Profit Margin (%)'] = (region_stats['Profit'] / region_stats['Sales'] * 100).round(2)
    region_stats = region_stats.sort_values('Sales', ascending=False)
    
    return region_stats

# %%
# Main execution for testing
if __name__ == "__main__":
    # Test the data preparation module
    print("Testing Data Preparation Module...")
    print("\n" + "="*25)
    
    # Initialize preprocessor (update path as needed)
    preprocessor = DataPreprocessor('global_superstore.csv')
    
    # Load data
    preprocessor.load_data()
    
    # Explore initial stats
    preprocessor.explore_initial_stats()
    
    # Clean data
    cleaned_df = preprocessor.clean_data()
    
    # Validate
    preprocessor.validate_data()
    
    # Get summary statistics
    summary = preprocessor.get_summary_stats()
    print(f"\nSummary Statistics:")
    for key, value in summary.items():
        if 'sales' in key or 'profit' in key:
            print(f"  {key}: ${value:,.2f}")
        else:
            print(f"  {key}: {value:,.0f}")
    
    # Save cleaned data
    preprocessor.save_cleaned_data('cleaned_superstore.csv')
    
    # Test helper functions
    print("\n" + "="*50)
    print("TESTING HELPER FUNCTIONS")
    print("="*50)
    
    top_5 = get_top_customers(cleaned_df)
    print("\nTop 5 Customers by Sales:")
    print(top_5.to_string(index=False))
    
    segment_perf = get_segment_performance(cleaned_df)
    print("\nSegment Performance:")
    print(segment_perf.to_string(index=False))
    
    print("\nData preparation module is ready for use!")


