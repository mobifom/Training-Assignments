import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ============================================================
# SIMPLE K-FOLD CROSS-VALIDATION EXAMPLE
# ============================================================

print("=" * 60)
print("K-FOLD CROSS-VALIDATION: SIMPLE EXAMPLE")
print("=" * 60)

# Step 1: Load the data
iris = load_iris()
X, y = iris.data, iris.target

print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Classes: {iris.target_names}")

# Step 2: Create our model
model = RandomForestClassifier(n_estimators=50, random_state=42)

# Step 3: Set up K-Fold (let's use k=5)
k = 5
kfold = KFold(n_splits=k, shuffle=True, random_state=42)

print(f"\nUsing {k}-Fold Cross-Validation")
print("=" * 40)

# ============================================================
# MANUAL K-FOLD DEMONSTRATION (to see what happens inside)
# ============================================================

fold_results = []
fold_details = []

print("What happens in each fold:")
print("-" * 60)

fold_num = 1
for train_index, test_index in kfold.split(X):
    # Split the data
    X_train_fold, X_test_fold = X[train_index], X[test_index]
    y_train_fold, y_test_fold = y[train_index], y[test_index]
    
    # Train the model on this fold's training data
    model_fold = RandomForestClassifier(n_estimators=50, random_state=42)
    model_fold.fit(X_train_fold, y_train_fold)
    
    # Test on this fold's test data
    y_pred_fold = model_fold.predict(X_test_fold)
    accuracy = accuracy_score(y_test_fold, y_pred_fold)
    
    # Store results
    fold_results.append(accuracy)
    fold_details.append({
        'fold': fold_num,
        'train_size': len(train_index),
        'test_size': len(test_index),
        'accuracy': accuracy,
        'train_indices': train_index,
        'test_indices': test_index
    })
    
    print(f"Fold {fold_num}: Train on {len(train_index)} samples, Test on {len(test_index)} samples → Accuracy: {accuracy:.3f}")
    fold_num += 1

# Calculate final cross-validation score
cv_mean = np.mean(fold_results)
cv_std = np.std(fold_results)

print("\n" + "=" * 40)
print("FINAL RESULTS:")
print(f"Individual fold scores: {[f'{score:.3f}' for score in fold_results]}")
print(f"Average CV Accuracy: {cv_mean:.3f} (±{cv_std:.3f})")
print("=" * 40)

# ============================================================
# COMPARE WITH SKLEARN'S CROSS_VAL_SCORE
# ============================================================

print("\nVerification using sklearn's cross_val_score:")
sklearn_scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
sklearn_mean = sklearn_scores.mean()

print(f"Sklearn CV scores: {[f'{score:.3f}' for score in sklearn_scores]}")
print(f"Sklearn mean: {sklearn_mean:.3f}")
print(f"Match our manual calculation: {np.allclose(fold_results, sklearn_scores)}")

# ============================================================
# VISUALIZATION
# ============================================================

# Create comprehensive visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Show how data is split in each fold
ax1 = axes[0, 0]

# Create a visual representation of the data splits
n_samples = len(X)
colors = ['lightblue', 'lightcoral']
labels = ['Training', 'Testing']

y_positions = []
fold_labels = []

for i, fold_info in enumerate(fold_details):
    # Create arrays to show train/test split
    split_array = np.zeros(n_samples)
    split_array[fold_info['test_indices']] = 1  # 1 for test, 0 for train
    
    # Plot this fold
    y_pos = i
    y_positions.append(y_pos)
    fold_labels.append(f"Fold {fold_info['fold']}")
    
    # Color code the samples
    for j, is_test in enumerate(split_array):
        color = colors[int(is_test)]
        ax1.barh(y_pos, 1, left=j, height=0.8, color=color, edgecolor='white', linewidth=0.5)

ax1.set_yticks(y_positions)
ax1.set_yticklabels(fold_labels)
ax1.set_xlabel('Sample Index')
ax1.set_title('K-Fold Data Splits Visualization')
ax1.legend(handles=[plt.Rectangle((0,0),1,1, color=colors[0], label='Training'),
                   plt.Rectangle((0,0),1,1, color=colors[1], label='Testing')],
          loc='upper right')

# Plot 2: Accuracy scores for each fold
ax2 = axes[0, 1]
fold_numbers = [f"Fold {i+1}" for i in range(k)]
bars = ax2.bar(fold_numbers, fold_results, color=['skyblue', 'lightgreen', 'gold', 'lightcoral', 'plum'])
ax2.axhline(y=cv_mean, color='red', linestyle='--', label=f'Mean: {cv_mean:.3f}')
ax2.set_ylabel('Accuracy')
ax2.set_title('Accuracy Score for Each Fold')
ax2.set_ylim(0.8, 1.0)
ax2.legend()

# Add value labels on bars
for i, (bar, score) in enumerate(zip(bars, fold_results)):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{score:.3f}', ha='center', va='bottom')

# Plot 3: Train vs Test sizes
ax3 = axes[1, 0]
train_sizes = [fold['train_size'] for fold in fold_details]
test_sizes = [fold['test_size'] for fold in fold_details]

x = np.arange(len(fold_numbers))
width = 0.35

ax3.bar(x - width/2, train_sizes, width, label='Training Size', color='lightblue')
ax3.bar(x + width/2, test_sizes, width, label='Testing Size', color='lightcoral')

ax3.set_xlabel('Fold')
ax3.set_ylabel('Number of Samples')
ax3.set_title('Training vs Testing Set Sizes')
ax3.set_xticks(x)
ax3.set_xticklabels(fold_numbers)
ax3.legend()

# Plot 4: Comparison with single train-test split
ax4 = axes[1, 1]

# Single train-test split for comparison
from sklearn.model_selection import train_test_split
X_train_single, X_test_single, y_train_single, y_test_single = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model_single = RandomForestClassifier(n_estimators=50, random_state=42)
model_single.fit(X_train_single, y_train_single)
single_accuracy = model_single.score(X_test_single, y_test_single)

# Create comparison chart
methods = ['Single\nTrain-Test', 'K-Fold CV\n(Average)']
accuracies = [single_accuracy, cv_mean]
errors = [0, cv_std]  # No error bar for single split

bars = ax4.bar(methods, accuracies, yerr=errors, capsize=5, 
               color=['orange', 'green'], alpha=0.7)
ax4.set_ylabel('Accuracy')
ax4.set_title('Single Split vs K-Fold Cross-Validation')
ax4.set_ylim(0.8, 1.0)

# Add value labels
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{acc:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# ============================================================
# SIMPLE ALGORITHM EXPLANATION
# ============================================================

print("\n" + "=" * 60)
print("K-FOLD ALGORITHM EXPLANATION")
print("=" * 60)

print("""
🔍 WHAT IS K-FOLD CROSS-VALIDATION?

Think of it like testing a student with multiple practice exams:
- Instead of one big test, give 5 smaller tests
- Each test uses different questions
- Average the scores for final grade

📚 THE ALGORITHM (Simple Steps):

1. SPLIT THE DATA:
   • Divide your dataset into K equal parts (folds)
   • Like cutting a pizza into 5 slices 🍕

2. FOR EACH FOLD:
   • Use 4 slices for TRAINING the model
   • Use 1 slice for TESTING the model
   • Record the test score

3. REPEAT:
   • Do this K times (5 times in our example)
   • Each time, use a different slice for testing

4. AVERAGE:
   • Take the average of all 5 test scores
   • This is your final cross-validation score

🎯 WHY IS THIS BETTER?

Single Test:
❌ What if your test data was too easy/hard?
❌ Results might be misleading

K-Fold Cross-Validation:
✅ Tests on multiple different data pieces
✅ More reliable and robust results
✅ Reduces chance of getting lucky/unlucky

🔢 IN OUR EXAMPLE:
""")

print(f"• Total samples: {len(X)}")
print(f"• K folds: {k}")
print(f"• Each fold tests on ~{len(X)//k} samples")
print(f"• Each fold trains on ~{len(X) - len(X)//k} samples")
print(f"• Final reliability: {cv_mean:.3f} ± {cv_std:.3f}")

print(f"""
🏆 INTERPRETATION:
- Our model achieves {cv_mean:.1%} accuracy on average
- The ± {cv_std:.3f} shows how consistent it is across different data splits
- Lower standard deviation = more reliable model

💡 KEY INSIGHT:
Cross-validation helps answer: "Will my model work well on NEW, unseen data?"
""")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print("✅ K-Fold Cross-Validation completed successfully!")
print(f"✅ Model tested on {k} different data combinations")
print(f"✅ Average performance: {cv_mean:.1%}")
print(f"✅ Performance consistency: ±{cv_std:.3f}")
print("✅ More reliable than single train-test split!")

# Show the difference in reliability
reliability_improvement = abs(single_accuracy - cv_mean)
print(f"\n📊 Single split accuracy: {single_accuracy:.3f}")
print(f"📊 K-fold average accuracy: {cv_mean:.3f}")
if reliability_improvement > 0.01:
    print(f"⚠️  Difference of {reliability_improvement:.3f} shows why CV is important!")
else:
    print("✅ Results are consistent - good sign!")