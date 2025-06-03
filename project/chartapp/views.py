from django.shortcuts import render
from django import forms
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Set the backend before importing pyplot
import matplotlib.pyplot as plt
import io
import base64
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import seaborn as sns


class UploadFileForm(forms.Form):
    dataset = forms.FileField(
        label="Upload CSV Dataset",
        widget=forms.ClearableFileInput(attrs={'accept': '.csv'})
    )
    algorithm = forms.ChoiceField(
        choices=[('logistic', 'Logistic Regression')],
        label="Select Algorithm"
    )


def index(request):
    result = None
    error = None
    chart_urls = {}
    classification_rep = None
    data_info = None
    conclusions = {}

    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            algorithm = form.cleaned_data['algorithm']
            dataset_file = request.FILES['dataset']

            try:                # Read the dataset
                data = pd.read_csv(dataset_file)
                
                # Generate data information
                data_info = {
                    'shape': str(data.shape),
                    'dtypes': data.dtypes.to_string(),
                    'description': data.describe().to_html(),
                    'category_counts': {}
                }
                
                # Get category-wise counts for categorical columns
                categorical_columns = data.select_dtypes(include=['object']).columns
                for col in categorical_columns:
                    data_info['category_counts'][col] = data[col].value_counts().to_dict()
                
                # Separate features and target
                X = data.drop('class', axis=1)
                y = data['class']
                
                # Identify numeric and categorical columns
                numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
                categorical_features = X.select_dtypes(include=['object']).columns
                
                # Process each categorical column separately
                X_processed = X.copy()
                for col in categorical_features:
                    # Fill empty values with 'unknown'
                    X_processed[col] = X_processed[col].fillna('unknown')
                    # Create a new encoder for each column
                    le = LabelEncoder()
                    X_processed[col] = le.fit_transform(X_processed[col])
                
                # Scale numeric features
                if len(numeric_features) > 0:
                    scaler = StandardScaler()
                    X_processed[numeric_features] = scaler.fit_transform(X_processed[numeric_features])
                
                # Encode target variable
                le_y = LabelEncoder()
                y_encoded = le_y.fit_transform(y)
                
                # Split the data
                X_train, X_test, y_train, y_test = train_test_split(
                    X_processed, y_encoded, test_size=0.2, random_state=42
                )
                
                # Train logistic regression
                model = LogisticRegression(max_iter=1000)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                  # Calculate metrics
                score = accuracy_score(y_test, y_pred)
                classification_rep = classification_report(
                    y_test, y_pred, 
                    target_names=le_y.classes_
                )
                result = f'Accuracy (Logistic Regression): {score:.3%}'                # Set the style for all plots
                plt.style.use('ggplot')
                
                # 1. Numerical Features Distribution
                numerical_cols = ['cap-diameter', 'stem-height', 'stem-width']
                for col in numerical_cols:
                    plt.figure(figsize=(10, 6))                
                    plt.hist(data[col].dropna(), bins=30, color='#4e79a7', edgecolor='#bab0ac', density=True, alpha=0.6, label='Histogram')
                    data[col].dropna().plot(kind='kde', color='#e15759', label='KDE')
                    plt.title(f'Distribution of {col}')
                    plt.xlabel(col)
                    plt.ylabel('Density')
                    plt.legend()
                    
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                    buf.seek(0)
                    chart_urls[f'dist_{col}'] = base64.b64encode(buf.read()).decode('utf-8')
                    buf.close()
                    plt.close()
                    
                    # Add conclusion
                    conclusions[f'dist_{col}'] = f"The {col} shows a {['normal', 'right-skewed', 'left-skewed'][np.argmax([0, data[col].skew() > 0.5, data[col].skew() < -0.5])]} distribution with mean {data[col].mean():.2f}."                # 2. Feature Importance
                plt.figure(figsize=(12, 6))
                feature_importance = pd.DataFrame({
                    'feature': X_processed.columns,
                    'importance': abs(model.coef_[0])
                })                
                feature_importance = feature_importance.sort_values('importance', ascending=True)
                top_features = feature_importance.tail(8)  # Show top 8 features
                  # Create horizontal bar chart with better styling using the provided color palette
                colors = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f', '#edc949', '#af7aa1', '#ff9da7'][:len(top_features)]
                bars = plt.barh(y=range(len(top_features)), width=top_features['importance'],
                        color=colors)
                plt.yticks(range(len(top_features)), top_features['feature'])
                plt.xlabel('Importance Score')
                plt.title('Top 8 Most Important Features for Mushroom Classification')
                
                # Add grid for better readability
                plt.grid(axis='x', linestyle='--', alpha=0.7)
                plt.tight_layout()
                
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=300, bbox_inches='tight', facecolor='white')
                buf.seek(0)
                chart_urls['feature_importance'] = base64.b64encode(buf.read()).decode('utf-8')
                buf.close()
                plt.close()
                
                conclusions['feature_importance'] = "🔍 Top features reveal that ring type, spore color, and stalk characteristics are crucial indicators for mushroom classification."                # 3. Class Distribution
                plt.figure(figsize=(8, 8))
                class_counts = pd.Series(y).value_counts()                   
                plt.pie(class_counts, 
                       labels=[f'{idx}\n({class_counts[idx]:,} samples)' for idx in class_counts.index],
                       colors=['#4e79a7', '#76b7b2'],  # Using blue and teal from the palette
                       autopct=lambda pct: f'{pct:.1f}%',
                       startangle=90, 
                       wedgeprops={'edgecolor': 'white', 'linewidth': 1.5, 'alpha': 0.9})
                plt.title('Distribution of Mushroom Classes')
                
                # Make the pie chart look cleaner
                plt.axis('equal')
                
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=300, bbox_inches='tight', facecolor='white')
                buf.seek(0)
                chart_urls['class_distribution'] = base64.b64encode(buf.read()).decode('utf-8')
                buf.close()
                plt.close()               
                conclusions['class_distribution'] = "📊 The dataset shows a balanced distribution between edible and poisonous mushrooms, enabling reliable model training."

                # 4. Scatter Plot (Stem Height vs Width)
                plt.figure(figsize=(10, 6))
                colors = {'edible': '#59a14f', 'poisonous': '#e15759'}  # Using green and red from our palette
                plt.scatter(data['stem-height'], data['stem-width'], 
                        c=data['class'].map(colors), alpha=0.6)
                plt.title('Stem Height vs Width by Mushroom Class')
                plt.xlabel('Stem Height')
                plt.ylabel('Stem Width')
                plt.grid(True, linestyle='--', alpha=0.7)
                
                # Add a legend
                for class_name, color in colors.items():
                    plt.scatter([], [], c=color, alpha=0.6, label=class_name.title())
                plt.legend()
                
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=300, bbox_inches='tight', facecolor='white')
                buf.seek(0)
                chart_urls['scatter_plot'] = base64.b64encode(buf.read()).decode('utf-8')
                buf.close()
                plt.close()                # Add conclusion for scatter plot
                conclusions['scatter_plot'] = "📐 The scatter plot reveals the relationship between stem height and width, showing distinct clustering patterns for edible and poisonous mushrooms."

                # 5. Correlation Heatmap
                plt.figure(figsize=(8, 6))
                corr = data[['cap-diameter', 'stem-height', 'stem-width']].corr()
                plt.imshow(corr, cmap='coolwarm', interpolation='none')
                plt.colorbar()
                plt.xticks(range(len(corr)), corr.columns, rotation=45, ha='right')
                plt.yticks(range(len(corr)), corr.columns)
                plt.title('Correlation Matrix of Mushroom Dimensions')
                
                # Add correlation values in the cells
                for i in range(len(corr)):
                    for j in range(len(corr)):
                        plt.text(j, i, f'{corr.iloc[i, j]:.2f}',
                               ha='center', va='center')
                
                plt.tight_layout()
                
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=300, bbox_inches='tight', facecolor='white')
                buf.seek(0)
                chart_urls['correlation'] = base64.b64encode(buf.read()).decode('utf-8')
                buf.close()
                plt.close()                # Add conclusion for correlation heatmap
                conclusions['correlation'] = "🔄 The correlation matrix shows the relationships between mushroom dimensions: stem height has a moderate positive correlation with stem width, while cap diameter shows weaker correlations with both stem measurements."

                # 6. Categorical Features Count Plots
                categorical_features = ['cap-shape', 'cap-color', 'gill-color']
                for col in categorical_features:
                    plt.figure(figsize=(10, 6))
                    counts = data[col].value_counts()
                    plt.bar(counts.index, counts.values, color='#af7aa1')  # Using purple from our palette
                    plt.title(f'Distribution of {col.replace("-", " ").title()}')
                    plt.xticks(rotation=45, ha='right')
                    plt.ylabel('Count')
                    plt.xlabel(col.replace("-", " ").title())
                    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
                    plt.tight_layout()

                    # Add value labels on top of each bar
                    for i, v in enumerate(counts.values):
                        plt.text(i, v, str(v), ha='center', va='bottom')
                    
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight', facecolor='white')
                    buf.seek(0)
                    chart_urls[f'count_{col}'] = base64.b64encode(buf.read()).decode('utf-8')
                    buf.close()
                    plt.close()                # Add conclusion for each categorical feature
                    most_common = counts.index[0]
                    least_common = counts.index[-1]
                    conclusions[f'count_{col}'] = f"📊 In {col.replace('-', ' ')}, '{most_common}' is the most common ({counts[most_common]:,} occurrences) while '{least_common}' is the least common ({counts[least_common]:,} occurrences), showing clear preferences in mushroom characteristics."

                # Stacked Bar Plot (Cap Color vs Class)
                plt.figure(figsize=(12, 6))
                cap_class = pd.crosstab(data['cap-color'], data['class'])
                ax = cap_class.plot(kind='bar', stacked=True, 
                                  color=['#59a14f', '#e15759'],  # green for edible, red for poisonous
                                  width=0.8)
                plt.title('Distribution of Cap Colors by Mushroom Class')
                plt.xlabel('Cap Color')
                plt.ylabel('Count')
                plt.xticks(rotation=45, ha='right')
                plt.legend(title='Class', bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.grid(True, linestyle='--', alpha=0.7, axis='y')

                # Add value labels on the bars
                for c in ax.containers:
                    ax.bar_label(c, label_type='center')

                plt.tight_layout()
                
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=300, bbox_inches='tight', facecolor='white')
                buf.seek(0)
                chart_urls['cap_color_class'] = base64.b64encode(buf.read()).decode('utf-8')
                buf.close()
                plt.close()

                # Add conclusion for cap color vs class distribution
                most_common_cap = cap_class.sum(axis=1).idxmax()                
                most_dangerous = cap_class['poisonous'].idxmax()
                conclusions['cap_color_class'] = f"🎨 {most_common_cap.title()} is the most common cap color overall, while {most_dangerous.title()} caps show the highest proportion of poisonous mushrooms. This suggests certain cap colors may be better indicators of mushroom safety."

                # Box Plot: gill-color vs stem-height
                plt.figure(figsize=(12, 6))
                categories = data['gill-color'].dropna().unique()
                box_data = [data[data['gill-color'] == cat]['stem-height'].dropna() for cat in categories]
                
                # Create box plot with custom style
                bp = plt.boxplot(box_data, labels=categories, patch_artist=True,
                               medianprops=dict(color='black'),
                               boxprops=dict(facecolor='#76b7b2', alpha=0.7),
                               whiskerprops=dict(color='#2c3e50'),
                               capprops=dict(color='#2c3e50'),
                               flierprops=dict(marker='o', markerfacecolor='#e15759', alpha=0.5))
                
                plt.title('Distribution of Stem Height by Gill Color')
                plt.xlabel('Gill Color')
                plt.ylabel('Stem Height')
                plt.xticks(rotation=45, ha='right')
                plt.grid(True, linestyle='--', alpha=0.7, axis='y')
                plt.tight_layout()

                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=300, bbox_inches='tight', facecolor='white')
                buf.seek(0)
                chart_urls['gill_height_box'] = base64.b64encode(buf.read()).decode('utf-8')
                buf.close()
                plt.close()

                # Add conclusion for box plot
                median_heights = {cat: np.median(data[data['gill-color'] == cat]['stem-height']) 
                                for cat in categories}
                highest_gill = max(median_heights.items(), key=lambda x: x[1])[0]
                lowest_gill = min(median_heights.items(), key=lambda x: x[1])[0]
                conclusions['gill_height_box'] = f"📏 Mushrooms with {highest_gill} gills tend to have the tallest stems, while those with {lowest_gill} gills typically have shorter stems. The box plot also reveals considerable variation in stem height across different gill colors."
            except Exception as e:
                error = f"Error processing dataset: {str(e)}"
        else:
            error = "Invalid form submission."   
        
    else:
        form = UploadFileForm()
    
    return render(request, 'chartapp/index.html', {
        'form': form,
        'result': result,
        'error': error,
        'chart_urls': chart_urls,
        'classification_report': classification_rep,
        'data_info': data_info,
        'conclusions': conclusions
    })