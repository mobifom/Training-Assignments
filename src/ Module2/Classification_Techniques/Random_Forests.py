import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load Iris dataset (multiclass classification)
iris = load_iris()
X, y = iris.data, iris.target

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Random Forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predictions
y_pred = rf.predict(X_test)

# Evaluate performance
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))

# -------------------------------
# Visualization (2D projection)
# -------------------------------
plt.figure(figsize=(8, 6))

# Use only first two features for plotting
scatter = plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap="viridis", edgecolor="k")

# Add legend manually to avoid ambiguity
for i, target_name in enumerate(iris.target_names):
    plt.scatter([], [], c=scatter.cmap(scatter.norm(i)), label=target_name)

plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title("Random Forest Classification on Iris (Test Data)")
plt.legend(title="Classes")
plt.show()


"""
ALGORITHM RandomForestClassification
INPUT: 
    - iris_dataset
    - test_size = 0.3
    - n_estimators = 100
    - random_state = 42

OUTPUT:
    - accuracy_score
    - classification_report
    - visualization_plot

BEGIN
    // Step 1: Data Preparation
    X, y = LOAD_IRIS_DATASET()
    X_train, X_test, y_train, y_test = SPLIT_DATA(X, y, test_size, random_state)
    
    // Step 2: Train Random Forest Model
    rf_model = TRAIN_RANDOM_FOREST(X_train, y_train, n_estimators, random_state)
    
    // Step 3: Make Predictions
    y_pred = PREDICT(rf_model, X_test)
    
    // Step 4: Evaluate Performance
    accuracy = CALCULATE_ACCURACY(y_test, y_pred)
    report = GENERATE_CLASSIFICATION_REPORT(y_test, y_pred)
    
    // Step 5: Create Visualization
    VISUALIZE_RESULTS(X_test, y_pred, iris.feature_names, iris.target_names)
    
    RETURN accuracy, report
END

FUNCTION SPLIT_DATA(X, y, test_size, random_state)
BEGIN
    SET random_seed = random_state
    n_samples = LENGTH(X)
    n_test = FLOOR(n_samples × test_size)
    n_train = n_samples - n_test
    
    indices = SHUFFLE([0, 1, 2, ..., n_samples-1], random_seed)
    
    train_indices = indices[0 : n_train]
    test_indices = indices[n_train : n_samples]
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    RETURN X_train, X_test, y_train, y_test
END

FUNCTION TRAIN_RANDOM_FOREST(X_train, y_train, n_estimators, random_state)
BEGIN
    forest = EMPTY_LIST()
    n_features = NUMBER_OF_FEATURES(X_train)
    max_features = SQRT(n_features)  // Feature subsampling size
    
    FOR i = 0 TO n_estimators-1 DO
        // Create bootstrap sample
        bootstrap_X, bootstrap_y = BOOTSTRAP_SAMPLE(X_train, y_train, random_state + i)
        
        // Train individual decision tree
        tree = TRAIN_DECISION_TREE(bootstrap_X, bootstrap_y, max_features, random_state + i)
        
        ADD tree TO forest
    END FOR
    
    RETURN forest
END

FUNCTION BOOTSTRAP_SAMPLE(X, y, seed)
BEGIN
    SET random_seed = seed
    n_samples = LENGTH(X)
    
    // Sample with replacement
    bootstrap_indices = []
    FOR i = 0 TO n_samples-1 DO
        random_index = RANDOM_INTEGER(0, n_samples-1, random_seed)
        ADD random_index TO bootstrap_indices
    END FOR
    
    bootstrap_X = X[bootstrap_indices]
    bootstrap_y = y[bootstrap_indices]
    
    RETURN bootstrap_X, bootstrap_y
END

FUNCTION TRAIN_DECISION_TREE(X, y, max_features, seed)
BEGIN
    tree = CREATE_EMPTY_TREE()
    root_node = BUILD_TREE_RECURSIVE(X, y, max_features, seed, depth=0)
    tree.root = root_node
    
    RETURN tree
END

FUNCTION BUILD_TREE_RECURSIVE(X, y, max_features, seed, depth)
BEGIN
    // Stopping criteria
    IF ALL_SAME_CLASS(y) OR depth > MAX_DEPTH OR LENGTH(y) < MIN_SAMPLES THEN
        leaf_node = CREATE_LEAF_NODE(MAJORITY_CLASS(y))
        RETURN leaf_node
    END IF
    
    // Random feature selection
    available_features = [0, 1, 2, ..., NUMBER_OF_FEATURES(X)-1]
    selected_features = RANDOMLY_SELECT(available_features, max_features, seed)
    
    // Find best split
    best_feature, best_threshold = FIND_BEST_SPLIT(X, y, selected_features)
    
    // Create internal node
    node = CREATE_INTERNAL_NODE(best_feature, best_threshold)
    
    // Split data
    left_X, left_y = FILTER_DATA(X, y, WHERE X[:, best_feature] <= best_threshold)
    right_X, right_y = FILTER_DATA(X, y, WHERE X[:, best_feature] > best_threshold)
    
    // Recursive calls
    node.left = BUILD_TREE_RECURSIVE(left_X, left_y, max_features, seed, depth+1)
    node.right = BUILD_TREE_RECURSIVE(right_X, right_y, max_features, seed, depth+1)
    
    RETURN node
END

FUNCTION PREDICT(forest, X_test)
BEGIN
    predictions = []
    
    FOR each sample IN X_test DO
        tree_votes = []
        
        // Get prediction from each tree
        FOR each tree IN forest DO
            vote = PREDICT_SINGLE_TREE(tree, sample)
            ADD vote TO tree_votes
        END FOR
        
        // Majority voting
        final_prediction = MAJORITY_VOTE(tree_votes)
        ADD final_prediction TO predictions
    END FOR
    
    RETURN predictions
END

FUNCTION PREDICT_SINGLE_TREE(tree, sample)
BEGIN
    current_node = tree.root
    
    WHILE current_node IS NOT LEAF DO
        feature_value = sample[current_node.feature_index]
        
        IF feature_value <= current_node.threshold THEN
            current_node = current_node.left
        ELSE
            current_node = current_node.right
        END IF
    END WHILE
    
    RETURN current_node.class_label
END

FUNCTION CALCULATE_ACCURACY(y_true, y_pred)
BEGIN
    correct_predictions = 0
    total_predictions = LENGTH(y_true)
    
    FOR i = 0 TO total_predictions-1 DO
        IF y_true[i] == y_pred[i] THEN
            correct_predictions = correct_predictions + 1
        END IF
    END FOR
    
    accuracy = correct_predictions / total_predictions
    RETURN accuracy
END

FUNCTION GENERATE_CLASSIFICATION_REPORT(y_true, y_pred)
BEGIN
    FOR each class IN [0, 1, 2] DO  // Setosa, Versicolor, Virginica
        TP = COUNT_TRUE_POSITIVES(y_true, y_pred, class)
        FP = COUNT_FALSE_POSITIVES(y_true, y_pred, class)
        FN = COUNT_FALSE_NEGATIVES(y_true, y_pred, class)
        
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1_score = 2 × (precision × recall) / (precision + recall)
        
        PRINT class, precision, recall, f1_score
    END FOR
END

FUNCTION VISUALIZE_RESULTS(X_test, y_pred, feature_names, target_names)
BEGIN
    // Use only first two features for 2D plot
    x_coords = X_test[:, 0]  // First feature (sepal length)
    y_coords = X_test[:, 1]  // Second feature (sepal width)
    
    // Create scatter plot
    CREATE_FIGURE(width=8, height=6)
    
    FOR each point i IN X_test DO
        color = MAP_CLASS_TO_COLOR(y_pred[i])
        PLOT_POINT(x_coords[i], y_coords[i], color)
    END FOR
    
    // Add labels and legend
    SET_XLABEL(feature_names[0])
    SET_YLABEL(feature_names[1])
    SET_TITLE("Random Forest Classification on Iris (Test Data)")
    
    FOR each class IN [0, 1, 2] DO
        ADD_LEGEND_ENTRY(target_names[class], MAP_CLASS_TO_COLOR(class))
    END FOR
    
    DISPLAY_PLOT()
END

FUNCTION MAJORITY_VOTE(votes)
BEGIN
    vote_counts = COUNT_OCCURRENCES(votes)
    RETURN CLASS_WITH_MAXIMUM_COUNT(vote_counts)
END

FUNCTION FIND_BEST_SPLIT(X, y, selected_features)
BEGIN
    best_gini = INFINITY
    best_feature = None
    best_threshold = None
    
    FOR each feature IN selected_features DO
        unique_values = GET_UNIQUE_VALUES(X[:, feature])
        
        FOR each value IN unique_values DO
            gini = CALCULATE_GINI_AFTER_SPLIT(X, y, feature, value)
            
            IF gini < best_gini THEN
                best_gini = gini
                best_feature = feature
                best_threshold = value
            END IF
        END FOR
    END FOR
    
    RETURN best_feature, best_threshold
END
"""