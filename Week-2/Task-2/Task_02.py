import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

df = pd.read_csv('Mall_Customers.csv')

print(f" Dataset loaded successfully!")
print(f"  Shape: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"   Columns: {list(df.columns)}")
print(f"\n First 5 rows:")
print(df.head())
print(f"\nMissing Values:\n{df.isnull().sum()}")
print(f"\n Statistical Summary (Numerical):")
print(df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].describe())
print(f"\n Categorical Columns Analysis:")
print(f"\nGender Distribution:\n{df['Gender'].value_counts()}")
print(f"\nEducation Distribution:\n{df['Education '].value_counts()}")
print(f"\nMarital Status Distribution:\n{df['Marital Status'].value_counts()}")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
sns.histplot(df['Age'], bins=20, kde=True, ax=axes[0,0], color='skyblue')
axes[0,0].set_title('Age Distribution', fontsize=14)

sns.histplot(df['Annual Income (k$)'], bins=20, kde=True, ax=axes[0,1], color='salmon')
axes[0,1].set_title('Annual Income Distribution', fontsize=14)

sns.histplot(df['Spending Score (1-100)'], bins=20, kde=True, ax=axes[0,2], color='lightgreen')
axes[0,2].set_title('Spending Score Distribution', fontsize=14)

gender_counts = df['Gender'].value_counts()
axes[1,0].pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', 
              colors=['#FF6B6B', '#4ECDC4'], startangle=90)
axes[1,0].set_title('Gender Distribution', fontsize=14)

edu_counts = df['Education '].value_counts()
axes[1,1].bar(edu_counts.index, edu_counts.values, color='purple', alpha=0.7)
axes[1,1].set_title('Education Distribution', fontsize=14)
axes[1,1].set_xlabel('Education Level')
axes[1,1].set_ylabel('Count')
axes[1,1].tick_params(axis='x', rotation=45)

marital_counts = df['Marital Status'].value_counts()
axes[1,2].bar(marital_counts.index, marital_counts.values, color='orange', alpha=0.7)
axes[1,2].set_title('Marital Status Distribution', fontsize=14)
axes[1,2].set_xlabel('Marital Status')
axes[1,2].set_ylabel('Count')
axes[1,2].tick_params(axis='x', rotation=45)

plt.suptitle('Customer Demographics Analysis', fontsize=16, y=1.02)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
scatter = plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], 
                      c=df['Age'], cmap='viridis', s=50, alpha=0.7)
plt.colorbar(scatter, label='Age')
plt.title('Relationship: Annual Income vs Spending Score', fontsize=16)
plt.xlabel('Annual Income (k$)', fontsize=12)
plt.ylabel('Spending Score (1-100)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.show()

print("\n" + "="*60)
print(" STEP 3: K-MEANS CLUSTERING")
print("="*60)

X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--', linewidth=2, markersize=8, color='purple')
plt.title('Elbow Method For Optimal k', fontsize=16)
plt.xlabel('Number of Clusters (k)', fontsize=12)
plt.ylabel('Within-Cluster Sum of Squares (WCSS)', fontsize=12)
plt.xticks(range(1, 11))
plt.grid(True, alpha=0.3)
plt.show()

optimal_k = 5
print(f"\nOptimal number of clusters based on elbow: {optimal_k}")

kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
y_kmeans = kmeans.fit_predict(X)

df['Cluster'] = y_kmeans

centroids = kmeans.cluster_centers_
cluster_interpretations = {}

for i, centroid in enumerate(centroids):
    income = centroid[0]
    spending = centroid[1]
    
    if income > 80 and spending > 60:
        name = " High Income, High Spending - 'The Loyalists'"
    elif 40 <= income <= 80 and 40 <= spending <= 70:
        name = " Average Income, Average Spending - 'The Mainstream'"
    elif income < 40 and spending < 40:
        name = " Low Income, Low Spending - 'The Conservatives'"
    elif income > 80 and spending < 40:
        name = " High Income, Low Spending - 'The Untapped Potential'"
    elif income < 40 and spending > 60:
        name = " Low Income, High Spending - 'The Impulsives'"
    else:
        name = f"Cluster {i}"
    
    cluster_interpretations[i] = name

# Colors for clusters
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

plt.figure(figsize=(12, 8))
for i in range(optimal_k):
    plt.scatter(X[y_kmeans == i, 0], X[y_kmeans == i, 1], 
                s=80, c=colors[i], label=cluster_interpretations[i], alpha=0.7, 
                edgecolors='white', linewidth=1.5)

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            s=300, c='black', marker='X', label='Centroids', 
            edgecolors='white', linewidth=3)

plt.title('Customer Segments (K-Means Clustering)', fontsize=16)
plt.xlabel('Annual Income (k$)', fontsize=12)
plt.ylabel('Spending Score (1-100)', fontsize=12)
plt.legend(loc='best', fontsize=10, bbox_to_anchor=(1.05, 1))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\n" + "="*60)
print(" STEP 4: PCA DIMENSIONALITY REDUCTION")
print("="*60)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(12, 8))
for i in range(optimal_k):
    plt.scatter(X_pca[y_kmeans == i, 0], X_pca[y_kmeans == i, 1], 
                s=80, c=colors[i], label=cluster_interpretations[i], alpha=0.7)

plt.title('PCA Visualization of Customer Segments', fontsize=16)
plt.xlabel(f'First Principal Component ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=12)
plt.ylabel(f'Second Principal Component ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=12)
plt.legend(loc='best', fontsize=10, bbox_to_anchor=(1.05, 1))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"\n Explained Variance Ratio:")
print(f"   PC1 explains: {pca.explained_variance_ratio_[0]*100:.1f}% of variance")
print(f"   PC2 explains: {pca.explained_variance_ratio_[1]*100:.1f}% of variance")
print(f"   Total variance explained: {sum(pca.explained_variance_ratio_)*100:.1f}%")

print("\n" + "="*60)
print(" STEP 5: CLUSTER ANALYSIS & PROFILING")
print("="*60)


for i in range(optimal_k):
    cluster_data = df[df['Cluster'] == i]
    print(f"\n{'─'*60}")
    print(f" {cluster_interpretations[i]}")
    print(f"{'─'*60}")
    print(f"    Number of customers: {len(cluster_data)} ({len(cluster_data)/len(df)*100:.1f}%)")
    print(f"    Average Annual Income: ${cluster_data['Annual Income (k$)'].mean():.0f}k")
    print(f"    Average Spending Score: {cluster_data['Spending Score (1-100)'].mean():.0f}")
    print(f"    Average Age: {cluster_data['Age'].mean():.1f} years")
    print(f"    Gender: {cluster_data['Gender'].value_counts().index[0]} ({cluster_data['Gender'].value_counts().iloc[0]} customers)")
    print(f"    Common Education: {cluster_data['Education '].value_counts().index[0]}")
    print(f"    Common Marital Status: {cluster_data['Marital Status'].value_counts().index[0]}")

print("\n" + "="*60)
print(" STEP 6: MARKETING STRATEGIES")
print("="*60)

marketing_strategies = {
    'High Income, High Spending': {
        'profile': 'Wealthy customers who actively spend money at the mall',
        'strategy': 'PREMIUM LOYALTY & RETENTION',
        'tactics': [
            'Create exclusive VIP loyalty program with premium perks',
            'Offer personal shopper and concierge services',
            'Provide early access to new collections and sales',
            'Implement referral bonuses for bringing similar high-value customers',
            'Host exclusive events and fashion shows for this segment'
        ]
    },
    'Average Income, Average Spending': {
        'profile': 'Balanced customers representing the middle class segment',
        'strategy': 'UPSELLING & CROSS-SELLING',
        'tactics': [
            'Create product bundles with slight discounts',
            'Run seasonal sales events (holiday, back-to-school)',
            'Implement standard points-based loyalty program',
            'Send targeted email campaigns with personalized recommendations',
            'Offer buy-one-get-one promotions on popular items'
        ]
    },
    'Low Income, Low Spending': {
        'profile': 'Customers with limited financial resources and low consumption',
        'strategy': 'VALUE FOCUS & MINIMAL INVESTMENT',
        'tactics': [
            'Promote mass discount events and clearance sales',
            'Use loss leaders on essential items to drive foot traffic',
            'Send passive communication via standard email blasts only',
            'Focus marketing budget on higher-value segments',
            'Offer basic loyalty tier with minimal benefits'
        ]
    },
    'High Income, Low Spending': {
        'profile': 'Affluent customers who don\'t currently spend much - MAJOR OPPORTUNITY',
        'strategy': 'RE-ENGAGEMENT & EDUCATION',
        'tactics': [
            'Send targeted discovery campaigns with premium lookbooks',
            'Offer free samples and experiential marketing events',
            'Promote cause marketing (sustainability, charity initiatives)',
            'Provide exclusive preview invitations and product launches',
            'Implement education-based marketing about product quality'
        ]
    },
    'Low Income, High Spending': {
        'profile': 'Customers who love to spend, often beyond their means',
        'strategy': 'CONVERSION & PROMOTION-DRIVEN',
        'tactics': [
            'Run flash sales and limited-time offers to create urgency',
            'Integrate Buy Now, Pay Later (BNPL) services like Klarna, Afterpay',
            'Offer accelerated loyalty points (Double Points weekends)',
            'Send strategic discount codes and coupon marketing',
            'Implement gamified shopping experiences and rewards'
        ]
    }
}


for cluster_name, strategy in marketing_strategies.items():
    print(f"\n{'='*70}")
    print(f" {cluster_name}")
    print(f"{'='*70}")
    print(f"\n Customer Profile: {strategy['profile']}")
    print(f"\n Primary Strategy: {strategy['strategy']}")
    print(f"\n Recommended Tactics:")
    for tactic in strategy['tactics']:
        print(f"   {tactic}")
    print()


print("\n" + "="*60)
print(" CROSS-TABULATION ANALYSIS")
print("="*60)

print("\n Cluster vs Gender:")
cluster_gender = pd.crosstab(df['Cluster'], df['Gender'])
print(cluster_gender)

print("\n Cluster vs Education:")
cluster_education = pd.crosstab(df['Cluster'], df['Education '])
print(cluster_education)

print("\n Cluster vs Marital Status:")
cluster_marital = pd.crosstab(df['Cluster'], df['Marital Status'])
print(cluster_marital)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

cluster_gender.plot(kind='bar', ax=axes[0], color=['#FF6B6B', '#4ECDC4'])
axes[0].set_title('Cluster Distribution by Gender', fontsize=14)
axes[0].set_xlabel('Cluster')
axes[0].set_ylabel('Count')
axes[0].legend(title='Gender')

cluster_education.plot(kind='bar', ax=axes[1], colormap='viridis')
axes[1].set_title('Cluster Distribution by Education', fontsize=14)
axes[1].set_xlabel('Cluster')
axes[1].set_ylabel('Count')
axes[1].legend(title='Education', bbox_to_anchor=(1.05, 1))

cluster_marital.plot(kind='bar', ax=axes[2], colormap='plasma')
axes[2].set_title('Cluster Distribution by Marital Status', fontsize=14)
axes[2].set_xlabel('Cluster')
axes[2].set_ylabel('Count')
axes[2].legend(title='Marital Status', bbox_to_anchor=(1.05, 1))

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print(" EXECUTIVE SUMMARY")
print("="*60)

cluster_sizes = df['Cluster'].value_counts().sort_index()

print(f"""
 COMPLETED SUCCESSFULLY!

 KEY FINDINGS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Total Customers Analyzed: {len(df)}
• Number of Segments Identified: {optimal_k}
• Best K Value Determined: {optimal_k} (Elbow Method)

 SEGMENT BREAKDOWN:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━""")

for i in range(optimal_k):
    size = cluster_sizes[i] if i in cluster_sizes.index else 0
    percentage = (size/len(df))*100
    print(f"   Cluster {i} ({cluster_interpretations[i].split(' - ')[0]}): {size} customers ({percentage:.1f}%)")

print(f"""
       PRIORITY ORDER FOR MARKETING INVESTMENT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1 High Income, High Spending (The Loyalists)     → Highest ROI, focus on retention
2 High Income, Low Spending (Untapped Potential) → Largest growth opportunity  
3 Average Income, Average Spending (Mainstream)  → Volume drivers
4 Low Income, High Spending (Impulsives)         → Promotional targets
5 Low Income, Low Spending (Conservatives)       → Minimal investment

 NEXT STEPS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Deploy cluster labels to your CRM system
2. Design targeted campaigns for high-value segments first
3. A/B test promotional offers on impulsive buyers
4. Track segment movement monthly to optimize strategies
5. Use education and marital status insights for personalized marketing
 Analysis Complete! Results saved below.
""")

df.to_csv('customer_segments_results.csv', index=False)
print(f" Results saved to: customer_segments_results.csv")
print(f"   File includes original data + cluster labels")

print(f"\n Sample of Results (First 10 customers with clusters):")
print(df[['CustomerID', 'Gender', 'Annual Income (k$)', 'Spending Score (1-100)', 'Cluster']].head(10))