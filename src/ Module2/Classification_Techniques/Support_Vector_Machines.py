import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Generate a synthetic dataset
X, y = make_classification(
    n_samples=200, n_features=2, n_classes=2,
    n_informative=2, n_redundant=0, random_state=42
)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train an SVM with RBF kernel
clf = SVC(kernel="rbf", C=1.0, gamma="scale")
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Evaluate performance
print("SVM Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Plot decision boundary
xx, yy = np.meshgrid(
    np.linspace(X[:,0].min()-1, X[:,0].max()+1, 200),
    np.linspace(X[:,1].min()-1, X[:,1].max()+1, 200)
)
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.2, cmap=plt.cm.coolwarm)
plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.coolwarm, edgecolors="k")
plt.title("Support Vector Machine (SVM) Classification")
plt.show()

"""
Random Forest vs SVM Classification
Random Forest (Iris Dataset)

Algorithm: Ensemble of 100 decision trees using majority voting
Data: 3 classes (Setosa, Versicolor, Virginica iris species)
Features: 4 measurements (sepal/petal length & width)
Key Technique: Bootstrap sampling + random feature selection
Strengths: Handles overfitting well, works with mixed data types
Decision Boundary: Rectangular regions based on feature thresholds

SVM with RBF Kernel (Synthetic Dataset)

Algorithm: Finds optimal separating hyperplane with maximum margin
Data: 2 classes (binary classification)
Features: 2D synthetic data points
Key Technique: Kernel trick maps data to higher dimensions for non-linear separation
Strengths: Creates curved decision boundaries, memory efficient (stores only support vectors)
Decision Boundary: Smooth curves based on distance from support vectors
"""

"""
ALGORITHM SVMClassificationWithRBF
INPUT: 
    - n_samples = 200
    - n_features = 2
    - n_classes = 2
    - test_size = 0.3
    - kernel = "rbf"
    - C = 1.0
    - gamma = "scale"
    - random_state = 42

OUTPUT:
    - accuracy_score
    - classification_report
    - decision_boundary_plot

BEGIN
    // Step 1: Generate Synthetic Dataset
    X, y = GENERATE_SYNTHETIC_DATA(n_samples, n_features, n_classes, random_state)
    
    // Step 2: Split Data
    X_train, X_test, y_train, y_test = SPLIT_DATA(X, y, test_size, random_state)
    
    // Step 3: Train SVM Model
    svm_model = TRAIN_SVM_RBF(X_train, y_train, C, gamma)
    
    // Step 4: Make Predictions
    y_pred = PREDICT(svm_model, X_test)
    
    // Step 5: Evaluate Performance
    accuracy = CALCULATE_ACCURACY(y_test, y_pred)
    report = GENERATE_CLASSIFICATION_REPORT(y_test, y_pred)
    
    // Step 6: Visualize Decision Boundary
    VISUALIZE_DECISION_BOUNDARY(svm_model, X, y)
    
    RETURN accuracy, report
END

FUNCTION GENERATE_SYNTHETIC_DATA(n_samples, n_features, n_classes, random_state)
BEGIN
    SET random_seed = random_state
    
    // Generate cluster centers
    centers = RANDOMLY_GENERATE_CENTERS(n_classes, n_features, random_seed)
    
    // Generate samples around centers
    X = EMPTY_MATRIX(n_samples, n_features)
    y = EMPTY_ARRAY(n_samples)
    
    samples_per_class = n_samples / n_classes
    
    FOR class_idx = 0 TO n_classes-1 DO
        start_idx = class_idx × samples_per_class
        end_idx = start_idx + samples_per_class
        
        FOR i = start_idx TO end_idx-1 DO
            // Generate sample around class center with noise
            X[i] = centers[class_idx] + RANDOM_NORMAL_NOISE(n_features, random_seed)
            y[i] = class_idx
        END FOR
    END FOR
    
    // Add some complexity to make it interesting
    X = ADD_INFORMATIVE_FEATURES(X, n_features, random_seed)
    
    RETURN X, y
END

FUNCTION TRAIN_SVM_RBF(X_train, y_train, C, gamma)
BEGIN
    // Initialize SVM parameters
    IF gamma == "scale" THEN
        gamma_value = 1.0 / (n_features × VARIANCE(X_train))
    ELSE
        gamma_value = gamma
    END IF
    
    // Solve the SVM optimization problem
    svm_model = SOLVE_SVM_OPTIMIZATION(X_train, y_train, C, gamma_value)
    
    RETURN svm_model
END

FUNCTION SOLVE_SVM_OPTIMIZATION(X_train, y_train, C, gamma)
BEGIN
    n_samples = LENGTH(X_train)
    
    // Compute RBF kernel matrix
    K = COMPUTE_RBF_KERNEL_MATRIX(X_train, gamma)
    
    // Set up quadratic optimization problem
    // Minimize: (1/2) * α^T * Q * α - e^T * α
    // Subject to: 0 ≤ α_i ≤ C, Σ(α_i * y_i) = 0
    
    Q = COMPUTE_Q_MATRIX(K, y_train)
    e = ONES_VECTOR(n_samples)
    
    // Solve using Sequential Minimal Optimization (SMO) or similar
    alpha = SMO_SOLVER(Q, e, y_train, C)
    
    // Find support vectors (non-zero alphas)
    support_vectors = []
    support_vector_labels = []
    support_vector_alphas = []
    
    FOR i = 0 TO n_samples-1 DO
        IF alpha[i] > EPSILON THEN
            ADD X_train[i] TO support_vectors
            ADD y_train[i] TO support_vector_labels
            ADD alpha[i] TO support_vector_alphas
        END IF
    END FOR
    
    // Compute bias term
    bias = COMPUTE_BIAS(support_vectors, support_vector_labels, support_vector_alphas, gamma, C)
    
    svm_model = {
        support_vectors: support_vectors,
        support_vector_labels: support_vector_labels,
        support_vector_alphas: support_vector_alphas,
        bias: bias,
        gamma: gamma,
        C: C
    }
    
    RETURN svm_model
END

FUNCTION COMPUTE_RBF_KERNEL_MATRIX(X, gamma)
BEGIN
    n_samples = LENGTH(X)
    K = ZEROS_MATRIX(n_samples, n_samples)
    
    FOR i = 0 TO n_samples-1 DO
        FOR j = 0 TO n_samples-1 DO
            // RBF kernel: K(x_i, x_j) = exp(-γ * ||x_i - x_j||²)
            distance_squared = EUCLIDEAN_DISTANCE_SQUARED(X[i], X[j])
            K[i,j] = EXP(-gamma × distance_squared)
        END FOR
    END FOR
    
    RETURN K
END

FUNCTION EUCLIDEAN_DISTANCE_SQUARED(x1, x2)
BEGIN
    distance_sq = 0
    FOR k = 0 TO LENGTH(x1)-1 DO
        diff = x1[k] - x2[k]
        distance_sq = distance_sq + diff²
    END FOR
    
    RETURN distance_sq
END


FUNCTION SMO_SOLVER(Q, e, y, C)
BEGIN
    n_samples = LENGTH(y)
    alpha = ZEROS_ARRAY(n_samples)
    
    max_iterations = 1000
    tolerance = 1e-3
    
    FOR iteration = 0 TO max_iterations DO
        alpha_prev = COPY(alpha)
        
        FOR i = 0 TO n_samples-1 DO
            // Select second variable j ≠ i
            j = SELECT_SECOND_VARIABLE(i, alpha, y, Q)
            
            IF j == -1 THEN CONTINUE
            
            // Store old alphas
            alpha_i_old = alpha[i]
            alpha_j_old = alpha[j]
            
            // Compute bounds L and H
            L, H = COMPUTE_BOUNDS(alpha[i], alpha[j], y[i], y[j], C)
            
            IF L == H THEN CONTINUE
            
            // Compute eta (second derivative)
            eta = 2×Q[i,j] - Q[i,i] - Q[j,j]
            
            IF eta >= 0 THEN CONTINUE
            
            // Compute new alpha[j]
            alpha[j] = alpha[j] - (y[j] × (E[i] - E[j])) / eta
            alpha[j] = CLIP(alpha[j], L, H)
            
            IF ABS(alpha[j] - alpha_j_old) < tolerance THEN CONTINUE
            
            // Update alpha[i]
            alpha[i] = alpha[i] + y[i]×y[j]×(alpha_j_old - alpha[j])
        END FOR
        
        // Check convergence
        IF CONVERGED(alpha, alpha_prev, tolerance) THEN BREAK
    END FOR
    
    RETURN alpha
END

FUNCTION PREDICT(svm_model, X_test)
BEGIN
    predictions = []
    
    FOR each sample IN X_test DO
        decision_value = COMPUTE_DECISION_FUNCTION(svm_model, sample)
        
        IF decision_value >= 0 THEN
            prediction = 1
        ELSE
            prediction = 0  // or -1 depending on class encoding
        END IF
        
        ADD prediction TO predictions
    END FOR
    
    RETURN predictions
END

FUNCTION COMPUTE_DECISION_FUNCTION(svm_model, x)
BEGIN
    decision_value = svm_model.bias
    
    FOR i = 0 TO LENGTH(svm_model.support_vectors)-1 DO
        sv = svm_model.support_vectors[i]
        alpha = svm_model.support_vector_alphas[i]
        label = svm_model.support_vector_labels[i]
        
        // Compute RBF kernel between x and support vector
        distance_squared = EUCLIDEAN_DISTANCE_SQUARED(x, sv)
        kernel_value = EXP(-svm_model.gamma × distance_squared)
        
        decision_value = decision_value + alpha × label × kernel_value
    END FOR
    
    RETURN decision_value
END


FUNCTION VISUALIZE_DECISION_BOUNDARY(svm_model, X, y)
BEGIN
    // Create mesh grid for plotting
    x_min, x_max = MIN(X[:,0]) - 1, MAX(X[:,0]) + 1
    y_min, y_max = MIN(X[:,1]) - 1, MAX(X[:,1]) + 1
    
    xx, yy = CREATE_MESHGRID(x_min, x_max, y_min, y_max, step_size=0.02)
    
    // Flatten mesh grid for prediction
    grid_points = CONCATENATE_COLUMNS(FLATTEN(xx), FLATTEN(yy))
    
    // Predict for all grid points
    Z = PREDICT(svm_model, grid_points)
    Z = RESHAPE(Z, SHAPE(xx))
    
    // Create contour plot
    CREATE_FIGURE()
    
    // Plot decision regions
    PLOT_CONTOUR_FILLED(xx, yy, Z, alpha=0.2, colormap="coolwarm")
    
    // Plot original data points
    SCATTER_PLOT(X[:,0], X[:,1], colors=y, colormap="coolwarm", edge_colors="black")
    
    // Highlight support vectors if available
    IF svm_model.support_vectors IS NOT EMPTY THEN
        sv_x = svm_model.support_vectors[:,0]
        sv_y = svm_model.support_vectors[:,1]
        SCATTER_PLOT(sv_x, sv_y, marker="o", facecolor="none", 
                     edgecolor="black", linewidth=2, size=100)
    END IF
    
    SET_TITLE("Support Vector Machine (RBF Kernel) Classification")
    SET_XLABEL("Feature 1")
    SET_YLABEL("Feature 2")
    
    DISPLAY_PLOT()
END

FUNCTION CREATE_MESHGRID(x_min, x_max, y_min, y_max, step_size)
BEGIN
    x_range = LINSPACE(x_min, x_max, step_size)
    y_range = LINSPACE(y_min, y_max, step_size)
    xx, yy = MESHGRID(x_range, y_range)
    
    RETURN xx, yy
END

FUNCTION COMPUTE_BOUNDS(alpha_i, alpha_j, y_i, y_j, C)
BEGIN
    IF y_i == y_j THEN
        L = MAX(0, alpha_i + alpha_j - C)
        H = MIN(C, alpha_i + alpha_j)
    ELSE
        L = MAX(0, alpha_j - alpha_i)
        H = MIN(C, C + alpha_j - alpha_i)
    END IF
    
    RETURN L, H
END

FUNCTION COMPUTE_BIAS(support_vectors, labels, alphas, gamma, C)
BEGIN
    // Use support vectors that are not at bounds
    bias_sum = 0
    count = 0
    
    FOR i = 0 TO LENGTH(alphas)-1 DO
        IF 0 < alphas[i] < C THEN  // Not at bounds
            sv = support_vectors[i]
            label = labels[i]
            
            prediction_without_bias = 0
            FOR j = 0 TO LENGTH(support_vectors)-1 DO
                kernel_val = RBF_KERNEL(sv, support_vectors[j], gamma)
                prediction_without_bias += alphas[j] × labels[j] × kernel_val
            END FOR
            
            bias_sum += label - prediction_without_bias
            count += 1
        END IF
    END FOR
    
    IF count > 0 THEN
        RETURN bias_sum / count
    ELSE
        RETURN 0
    END IF
END


"""