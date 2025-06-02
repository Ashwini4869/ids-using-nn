import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer


def preprocess_data(train_df, test_df):
    """
    Preprocess the training and test datasets for the IDS model.

    Args:
        train_df (pd.DataFrame): Training dataset
        test_df (pd.DataFrame): Test dataset

    Returns:
        tuple: (X_train_processed, X_test_processed, y_train_encoded, y_test_encoded,
        preprocessor, feature_names)
    """

    # Define feature groups
    categorical_features = ["protocol_type", "service", "flag"]

    # Features requiring normalization (continuous with large ranges)
    continuous_features = [
        "duration",
        "src_bytes",
        "dst_bytes",
        "count",
        "srv_count",
        "dst_host_count",
        "dst_host_srv_count",
    ]

    # Features with small ranges (optional normalization)
    small_range_features = [
        "wrong_fragment",
        "urgent",
        "hot",
        "num_failed_logins",
        "num_compromised",
        "su_attempted",
        "num_root",
        "num_file_creations",
        "num_shells",
        "num_access_files",
    ]

    # Binary features (no preprocessing needed)
    binary_features = [
        "land",
        "logged_in",
        "is_host_login",
        "is_guest_login",
        "root_shell",
    ]

    # Rate-based features (already normalized)
    rate_features = [
        "serror_rate",
        "srv_serror_rate",
        "rerror_rate",
        "srv_rerror_rate",
        "same_srv_rate",
        "diff_srv_rate",
        "srv_diff_host_rate",
        "dst_host_same_srv_rate",
        "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate",
        "dst_host_serror_rate",
        "dst_host_srv_serror_rate",
        "dst_host_rerror_rate",
        "dst_host_srv_rerror_rate",
    ]

    # Drop constant column
    train_df = train_df.drop("num_outbound_cmds", axis=1)
    test_df = test_df.drop("num_outbound_cmds", axis=1)

    # Apply log transformation to bytes columns
    for df in [train_df, test_df]:
        df["src_bytes"] = np.log1p(df["src_bytes"])
        df["dst_bytes"] = np.log1p(df["dst_bytes"])

    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), continuous_features + small_range_features),
            ("cat", OneHotEncoder(drop="first"), categorical_features),
        ],
        remainder="passthrough",  # This will keep the binary and rate features as is
    )

    # Prepare data for preprocessing
    X_train = train_df.drop("class", axis=1)
    y_train = train_df["class"]
    X_test = test_df.drop("class", axis=1)
    y_test = test_df["class"]

    # Transform the features
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # Transform target variable
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)

    # Get feature names after preprocessing
    categorical_features_transformed = [
        f"{feat}_{val}"
        for i, feat in enumerate(categorical_features)
        for val in preprocessor.named_transformers_["cat"].categories_[i][1:]
    ]
    numeric_features = continuous_features + small_range_features
    passthrough_features = binary_features + rate_features

    feature_names = (
        numeric_features + categorical_features_transformed + passthrough_features
    )

    # Convert to DataFrame for better visualization
    X_train_processed = pd.DataFrame(X_train_transformed, columns=feature_names)
    X_test_processed = pd.DataFrame(X_test_transformed, columns=feature_names)

    return (
        X_train_processed,
        X_test_processed,
        y_train_encoded,
        y_test_encoded,
        preprocessor,
        feature_names,
    )
