import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from data_preparation import DataPreprocessor, get_top_customers, get_segment_performance

st.set_page_config(
    page_title="Global Superstore Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-title {
        font-size: 36px;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 20px;
    }
    .kpi-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
    .kpi-label {
        font-size: 16px;
        color: #666;
    }
    .kpi-value {
        font-size: 32px;
        font-weight: bold;
        color: #1E88E5;
    }
    </style>
""", unsafe_allow_html=True)

# Cache data loading for performance
@st.cache_data
def load_and_prepare_data(file_path):
    """
    Load and prepare the data with caching
    """
    preprocessor = DataPreprocessor(file_path)
    preprocessor.load_data()
    preprocessor.clean_data()
    return preprocessor.df

def display_kpi_metrics(df):
    """
    Display KPI cards for Total Sales, Profit, and other key metrics
    """
    col1, col2, col3, col4 = st.columns(4)
    
    total_sales = df['Sales'].sum()
    total_profit = df['Profit'].sum()
    profit_margin = (total_profit / total_sales * 100) if total_sales > 0 else 0
    total_orders = df['Order ID'].nunique()
    
    with col1:
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Total Sales</div>
                <div class="kpi-value">${total_sales:,.0f}</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        color = "#dc3545" if total_profit < 0 else "#28a745"
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Total Profit</div>
                <div class="kpi-value" style="color: {color};">${total_profit:,.0f}</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Profit Margin</div>
                <div class="kpi-value">{profit_margin:.1f}%</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Total Orders</div>
                <div class="kpi-value">{total_orders:,.0f}</div>
            </div>
        """, unsafe_allow_html=True)

def create_sales_trend_chart(df):
    """
    Create sales trend over time chart
    """
    if 'Order Date' in df.columns and not df['Order Date'].isna().all():
        # Aggregate by month
        df_monthly = df.groupby(df['Order Date'].dt.to_period('M')).agg({
            'Sales': 'sum',
            'Profit': 'sum'
        }).reset_index()
        df_monthly['Order Date'] = df_monthly['Order Date'].astype(str)
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Scatter(x=df_monthly['Order Date'], y=df_monthly['Sales'], 
                      name="Sales", line=dict(color='#1E88E5', width=2)),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(x=df_monthly['Order Date'], y=df_monthly['Profit'], 
                      name="Profit", line=dict(color='#28a745', width=2)),
            secondary_y=True
        )
        
        fig.update_layout(
            title="Sales & Profit Trend Over Time",
            xaxis_title="Date",
            hovermode='x unified',
            height=400
        )
        
        fig.update_yaxes(title_text="Sales ($)", secondary_y=False, color='#1E88E5')
        fig.update_yaxes(title_text="Profit ($)", secondary_y=True, color='#28a745')
        
        return fig
    return None

def create_segment_chart(df):
    """
    Create segment-wise performance chart
    """
    segment_data = df.groupby('Segment').agg({
        'Sales': 'sum',
        'Profit': 'sum'
    }).reset_index()
    
    fig = make_subplots(rows=1, cols=2, 
                        subplot_titles=('Sales by Segment', 'Profit by Segment'),
                        specs=[[{'type': 'bar'}, {'type': 'bar'}]])
    
    # Sales by segment
    fig.add_trace(
        go.Bar(x=segment_data['Segment'], y=segment_data['Sales'], 
               name='Sales', marker_color='#1E88E5'),
        row=1, col=1
    )
    
    # Profit by segment
    colors = ['#28a745' if x >= 0 else '#dc3545' for x in segment_data['Profit']]
    fig.add_trace(
        go.Bar(x=segment_data['Segment'], y=segment_data['Profit'], 
               name='Profit', marker_color=colors),
        row=1, col=2
    )
    
    fig.update_layout(height=400, showlegend=False, title_text="Segment Performance Analysis")
    fig.update_xaxes(title_text="Segment", row=1, col=1)
    fig.update_xaxes(title_text="Segment", row=1, col=2)
    fig.update_yaxes(title_text="Sales ($)", row=1, col=1)
    fig.update_yaxes(title_text="Profit ($)", row=1, col=2)
    
    return fig

def create_category_performance(df):
    """
    Create category and sub-category performance charts
    """
    # Category level
    category_data = df.groupby('Category').agg({
        'Sales': 'sum',
        'Profit': 'sum'
    }).reset_index()
    
    fig1 = px.bar(category_data, x='Category', y='Sales', 
                  title='Sales by Product Category',
                  color='Sales', color_continuous_scale='Blues',
                  text='Sales')
    fig1.update_traces(texttemplate='$%{text:.0f}', textposition='outside')
    
    # Sub-category top 10 by sales
    subcat_data = df.groupby('Sub-Category')['Sales'].sum().sort_values(ascending=False).head(10).reset_index()
    fig2 = px.bar(subcat_data, x='Sales', y='Sub-Category', 
                  title='Top 10 Sub-Categories by Sales',
                  orientation='h', color='Sales', 
                  color_continuous_scale='Greens')
    
    return fig1, fig2

def display_top_customers(df, n=5):
    """
    Display top N customers by sales
    """
    top_customers = get_top_customers(df, n)
    
    # Create a styled DataFrame for display
    display_df = top_customers.copy()
    display_df['Total Sales'] = display_df['Total Sales'].apply(lambda x: f"${x:,.0f}")
    display_df['Total Profit'] = display_df['Total Profit'].apply(lambda x: f"${x:,.0f}")
    
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Customer Name": "Customer",
            "Total Sales": st.column_config.TextColumn("Total Sales"),
            "Total Profit": st.column_config.TextColumn("Total Profit"),
            "Total Quantity": st.column_config.NumberColumn("Quantity"),
            "Order Count": st.column_config.NumberColumn("Orders")
        }
    )
    
    # Create a bar chart for top customers
    fig = px.bar(top_customers, x='Customer Name', y='Total Sales',
                 title=f'Top {n} Customers by Sales',
                 color='Total Sales', color_continuous_scale='Reds',
                 text='Total Sales')
    fig.update_traces(texttemplate='$%{text:.0f}', textposition='outside')
    fig.update_layout(xaxis_tickangle=-45, height=400)
    
    return fig

def main():
    """
    Main Streamlit application
    """
    # Title
    st.markdown('<div class="main-title">Global Superstore Interactive Dashboard</div>', 
                unsafe_allow_html=True)
    
    # Sidebar - File upload or default path
    st.sidebar.header("Data Source")
    
    # File uploader
    uploaded_file = st.sidebar.file_uploader("Upload Global Superstore CSV", type=['csv'])
    
    # Option to use default path
    use_default = st.sidebar.checkbox("Use default dataset path", value=True)
    
    # Load data
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with open('temp_superstore.csv', 'wb') as f:
            f.write(uploaded_file.getbuffer())
        df = load_and_prepare_data('temp_superstore.csv')
        st.sidebar.success("Data loaded from uploaded file!")
    elif use_default:
        df = load_and_prepare_data('global_superstore.csv')
        st.sidebar.info("Using default dataset: 'global_superstore.csv'")
    else:
        st.warning("Please upload a CSV file or use the default dataset path.")
        st.info("Download the Global Superstore dataset from: https://www.kaggle.com/datasets/apoorvaappz/global-super-store-dataset")
        st.stop()
    
    # Check if data loaded successfully
    if df is None or df.empty:
        st.error("Failed to load data. Please check the file path.")
        st.stop()
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Region filter
    regions = ['All'] + sorted(df['Region'].unique().tolist())
    selected_region = st.sidebar.selectbox("Select Region", regions)
    
    # Category filter
    categories = ['All'] + sorted(df['Category'].unique().tolist())
    selected_category = st.sidebar.selectbox("Select Category", categories)
    
    # Sub-category filter (dependent on category)
    if selected_category != 'All':
        subcategories = ['All'] + sorted(df[df['Category'] == selected_category]['Sub-Category'].unique().tolist())
    else:
        subcategories = ['All'] + sorted(df['Sub-Category'].unique().tolist())
    selected_subcategory = st.sidebar.selectbox("Select Sub-Category", subcategories)
    
    # Apply filters
    filtered_df = df.copy()
    
    if selected_region != 'All':
        filtered_df = filtered_df[filtered_df['Region'] == selected_region]
    
    if selected_category != 'All':
        filtered_df = filtered_df[filtered_df['Category'] == selected_category]
    
    if selected_subcategory != 'All':
        filtered_df = filtered_df[filtered_df['Sub-Category'] == selected_subcategory]
    
    # Check if filtered data is empty
    if filtered_df.empty:
        st.warning("No data available for the selected filters. Please adjust your selection.")
        st.stop()
    
    # Display KPIs
    st.header("Key Performance Indicators")
    display_kpi_metrics(filtered_df)
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Sales & Profit Trends", 
        "Segment Performance", 
        "Product Analysis", 
        "Top Customers",
        "Detailed Data"
    ])
    
    with tab1:
        st.header("Sales and Profit Analysis")
        
        # Time series chart
        if 'Order Date' in filtered_df.columns:
            trend_fig = create_sales_trend_chart(filtered_df)
            if trend_fig:
                st.plotly_chart(trend_fig, use_container_width=True)
            else:
                st.info("Insufficient date data for trend analysis")
        
        # Region performance
        st.subheader("Regional Performance")
        region_data = filtered_df.groupby('Region').agg({
            'Sales': 'sum',
            'Profit': 'sum'
        }).reset_index()
        
        col1, col2 = st.columns(2)
        with col1:
            fig_region_sales = px.bar(region_data, x='Region', y='Sales', 
                                      title='Sales by Region', color='Sales',
                                      color_continuous_scale='Blues')
            st.plotly_chart(fig_region_sales, use_container_width=True)
        
        with col2:
            fig_region_profit = px.bar(region_data, x='Region', y='Profit', 
                                       title='Profit by Region', color='Profit',
                                       color_continuous_scale='RdYlGn')
            st.plotly_chart(fig_region_profit, use_container_width=True)
    
    with tab2:
        st.header("Customer Segment Analysis")
        
        # Segment performance chart
        segment_fig = create_segment_chart(filtered_df)
        st.plotly_chart(segment_fig, use_container_width=True)
        
        # Detailed segment metrics
        st.subheader("Segment Performance Metrics")
        segment_metrics = filtered_df.groupby('Segment').agg({
            'Sales': 'sum',
            'Profit': 'sum',
            'Quantity': 'sum',
            'Order ID': 'count'
        }).round(2).reset_index()
        
        segment_metrics.columns = ['Segment', 'Total Sales', 'Total Profit', 'Total Quantity', 'Order Count']
        segment_metrics['Profit Margin (%)'] = (segment_metrics['Total Profit'] / segment_metrics['Total Sales'] * 100).round(2)
        
        st.dataframe(segment_metrics, use_container_width=True, hide_index=True)
        
        # Segment pie charts
        col1, col2 = st.columns(2)
        with col1:
            fig_pie_sales = px.pie(segment_metrics, values='Total Sales', names='Segment', 
                                   title='Sales Distribution by Segment', hole=0.3)
            st.plotly_chart(fig_pie_sales, use_container_width=True)
        
        with col2:
            fig_pie_profit = px.pie(segment_metrics, values='Total Profit', names='Segment', 
                                    title='Profit Distribution by Segment', hole=0.3)
            st.plotly_chart(fig_pie_profit, use_container_width=True)
    
    with tab3:
        st.header("Product Category & Sub-Category Analysis")
        
        # Category charts
        fig_cat, fig_subcat = create_category_performance(filtered_df)
        st.plotly_chart(fig_cat, use_container_width=True)
        st.plotly_chart(fig_subcat, use_container_width=True)
        
        # Profitability by sub-category
        st.subheader("Profitability Analysis by Sub-Category")
        subcat_profit = filtered_df.groupby('Sub-Category').agg({
            'Sales': 'sum',
            'Profit': 'sum'
        }).reset_index()
        subcat_profit['Profit Margin (%)'] = (subcat_profit['Profit'] / subcat_profit['Sales'] * 100).round(2)
        subcat_profit = subcat_profit.sort_values('Profit Margin (%)', ascending=False)
        
        fig_profit_margin = px.bar(subcat_profit.head(10), x='Sub-Category', y='Profit Margin (%)',
                                   title='Top 10 Sub-Categories by Profit Margin',
                                   color='Profit Margin (%)', color_continuous_scale='RdYlGn')
        st.plotly_chart(fig_profit_margin, use_container_width=True)
    
    with tab4:
        st.header("Top Customer Analysis")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            n_customers = st.number_input("Number of top customers to display", 
                                         min_value=5, max_value=20, value=5, step=5)
        
        # Display top customers
        top_customers_fig = display_top_customers(filtered_df, n_customers)
        st.plotly_chart(top_customers_fig, use_container_width=True)
        
        # Customer segmentation
        st.subheader("Customer Purchase Behavior")
        
        # Aggregate customer level data
        customer_data = filtered_df.groupby('Customer Name').agg({
            'Sales': 'sum',
            'Profit': 'sum',
            'Quantity': 'sum',
            'Order ID': 'count'
        }).reset_index()
        
        # Scatter plot: Sales vs Orders
        fig_scatter = px.scatter(customer_data, x='Order ID', y='Sales', 
                                size='Quantity', color='Profit',
                                title='Customer Analysis: Sales vs Number of Orders',
                                labels={'Order ID': 'Number of Orders', 'Sales': 'Total Sales ($)'},
                                color_continuous_scale='RdYlGn',
                                hover_data=['Customer Name'])
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with tab5:
        st.header("Detailed Data View")
        
        # Show filtered data
        st.subheader(f"Filtered Data ({len(filtered_df)} records)")
        
        # Select columns to display
        display_columns = ['Order Date', 'Region', 'Segment', 'Category', 'Sub-Category', 
                          'Customer Name', 'Sales', 'Profit', 'Quantity', 'Discount']
        available_cols = [col for col in display_columns if col in filtered_df.columns]
        
        st.dataframe(filtered_df[available_cols], use_container_width=True, height=400)
        
        # Download button
        csv = filtered_df[available_cols].to_csv(index=False)
        st.download_button(
            label="Download Filtered Data as CSV",
            data=csv,
            file_name="filtered_superstore_data.csv",
            mime="text/csv"
        )
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 20px;'>
        <p>Global Superstore Interactive Dashboard | Built with Streamlit, Plotly & Pandas</p>
        <p>Dataset: Global Superstore by apoorvaappz on Kaggle</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()