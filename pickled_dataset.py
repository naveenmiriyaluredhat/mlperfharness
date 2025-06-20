import pandas as pd
import pickle
import os

def create_dummy_pickled_df(file_path):
    """
    Creates a dummy pandas DataFrame and saves it as a pickled file.
    This is for demonstration purposes so the program can be run directly.
    In a real scenario, you would already have your .pkl file.
    """
    data = {
        'column_A': [1, 2, 3, 4],
        'column_B': ['apple', 'banana', 'cherry', 'date'],
        'column_C': [True, False, True, False]
    }
    df = pd.DataFrame(data)
    with open(file_path, 'wb') as f:
        pickle.dump(df, f)
    print(f"Dummy DataFrame pickled and saved to '{file_path}'")

def load_pickled_dataframe_and_print_header(file_path):
    """
    Loads a pickled pandas DataFrame from the given file path
    and prints its column headers.
    """
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' does not exist.")
        print("Please ensure the .pkl file is in the correct directory.")
        # Optionally, create a dummy file if it's missing for testing
        create_dummy_pickled_df(file_path)
        # Try loading again after creating the dummy file
        if not os.path.exists(file_path): # Check if dummy creation worked
            return

    try:
        # Load the DataFrame from the pickled file
        df = pd.read_pickle(file_path)

        print(f"\nSuccessfully loaded DataFrame from '{file_path}'.")
        print("\nDataFrame Header (Column Names):")
        # Print the column names (header) of the DataFrame
        print(df.columns.tolist())

        # Optionally, print the first few rows to confirm content
        print("\nFirst 5 rows of the DataFrame (for verification):")
        print(df.head())

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except pd.errors.EmptyDataError:
        print(f"Error: The file '{file_path}' is empty.")
    except (pickle.UnpicklingError, EOFError, AttributeError) as e:
        print(f"Error: Could not unpickle the file '{file_path}'. It might not be a valid pandas DataFrame pickle or is corrupted. Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Define the path to your pickled pandas DataFrame file
    # Replace 'your_dataframe.pkl' with the actual path to your file.
    # For demonstration, we'll create a dummy one if it doesn't exist.
    pickled_file_name = '../datasets/Llama2-70b/open_orca_gpt4_tokenized_llama.sampled_24576.pkl'

    # Ensure a dummy file exists for the first run if you don't have one
    if not os.path.exists(pickled_file_name):
        create_dummy_pickled_df(pickled_file_name)

    load_pickled_dataframe_and_print_header(pickled_file_name)

    # Example with a non-existent file (will try to create dummy first)
    # print("\n--- Testing with a non-existent file ---")
    # load_pickled_dataframe_and_print_header('non_existent_df.pkl')
