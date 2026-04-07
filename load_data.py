import pandas as pd

def load_data():
    print("Loading data...")
    # Load the datasets
    train_df = pd.read_csv("train.csv", low_memory=False)
    store_df = pd.read_csv("store.csv")
    test_df = pd.read_csv("test.csv")
    
    print("Data loaded successfully!")
    print(f"Train data shape: {train_df.shape}")
    print(f"Store data shape: {store_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    
    return train_df, store_df, test_df

if __name__ == "__main__":
    train, store, test = load_data()
    
    # Preview the data
    print("\nTrain Preview:")
    print(train.head())
