import pandas as pd


template = {
    "DUT SN": "",
    "Sys Type": "",
    "Test Group": "",
    "Test Name": "",
    "Board SN": "",
    "Chip Type": "",
    "Chip Num": "",
    "Channel": "",
    "PA": "",
    "Result": "",
    "Units": "",
    "Min Limit ATE": "",
    "Max Limit ATE": "",
    "Verdict ATE": "",
    "LOM Freq Config[MHz]": "",
    "RF Freq Config[MHz]": "",
    "BB Freq Config[MHz]": "",
    "Error Msg": "",
    "DTS[C]": "",
    "Iteration": "",
    "Chip ATE SN": "",
    "OTP ID": "",
    "OTP Version": "",
    "Digital Backoff": ""
}


df_combined = pd.DataFrame(columns=template.keys())

def reverse_conversion(tx=None, rx=None):
    """
    Conversion function from element to Chip, Channel in receiver and Chip, Channel, Pa in transmission.

    Parameters:
        tx (int): Transmission element index.
        rx (int): Receiver element index.

    Returns:
        Tuple: (chip_tx, channel_tx, pa, chip_rx, channel_rx)
        where chip_tx, channel_tx, pa are returned if tx is provided,
        and chip_rx, channel_rx are returned if rx is provided.
    """
    tx_dict = {
        0: [2, 3, 2], 1: [2, 3, 1], 2: [2, 2, 1], 3: [2, 2, 2], 4: [2, 1, 2],
        5: [2, 1, 1], 6: [2, 0, 1], 7: [2, 0, 2], 8: [1, 3, 2], 9: [1, 3, 1],
        10: [1, 2, 1], 11: [1, 2, 2], 12: [1, 1, 2], 13: [1, 1, 1], 14: [1, 0, 1],
        15: [1, 0, 2], 16: [0, 3, 2], 17: [0, 3, 1], 18: [0, 2, 1],
        19: [0, 2, 2], 20: [0, 1, 2], 21: [0, 1, 1], 22: [0, 0, 1], 23: [0, 0, 2],
        24: [5, 3, 2], 25: [5, 3, 1], 26: [5, 2, 1], 27: [5, 2, 2],
        28: [5, 1, 2], 29: [5, 1, 1], 30: [5, 0, 1], 31: [5, 0, 2], 32: [4, 3, 2],
        33: [4, 3, 1], 34: [4, 2, 1], 35: [4, 2, 2], 36: [4, 1, 2],
        37: [4, 1, 1], 38: [4, 0, 1], 39: [4, 0, 2], 40: [3, 3, 2], 41: [3, 3, 1],
        42: [3, 2, 1], 43: [3, 2, 2], 44: [3, 1, 2], 45: [3, 1, 1],
        46: [3, 0, 1], 47: [3, 0, 2]
    }

    rx_dict = {
        0: [0, 5], 1: [0, 4], 2: [0, 3], 3: [0, 2], 4: [0, 1], 5: [0, 0],
        6: [1, 5], 7: [1, 4], 8: [1, 3], 9: [1, 2], 10: [1, 1], 11: [1, 0],
        12: [2, 5], 13: [2, 4], 14: [2, 3], 15: [2, 2], 16: [2, 1], 17: [2, 0],
        18: [3, 5], 19: [3, 4], 20: [3, 3], 21: [3, 2], 22: [3, 1], 23: [3, 0],
        24: [4, 5], 25: [4, 4], 26: [4, 3], 27: [4, 2], 28: [4, 1], 29: [4, 0],
        30: [5, 5], 31: [5, 4], 32: [5, 3], 33: [5, 2], 34: [5, 1], 35: [5, 0],
        36: [6, 5], 37: [6, 4], 38: [6, 3], 39: [6, 2], 40: [6, 1], 41: [6, 0],
        42: [7, 5], 43: [7, 4], 44: [7, 3], 45: [7, 2], 46: [7, 1], 47: [7, 0]
    }

    if tx is not None:
        chip_tx, channel_tx, pa = tx_dict.get(tx, [None, None, None])
        return chip_tx, channel_tx, pa, None, None
    elif rx is not None:
        chip_rx, channel_rx = rx_dict.get(rx, [None, None])
        return None, None, None, chip_rx, channel_rx
    return None, None, None, None, None
def add_new_data(test_name, new_data, tx=None, rx=None):
    """
    Add new data to the combined DataFrame.

    Parameters:
        new_data (list): New data to be added. Each element is used as a value for "Result".
        tx (int or None): Transmission element index.
        rx (int or None): Receiver element index.
    """
    global df_combined

    df_new = pd.DataFrame(new_data, columns=["Result"])

    if tx:
        chip_nums, channels, pas, _, _ = zip(*[reverse_conversion(tx=idx) for idx in range(len(new_data))])
        chip_type = "Tx"
    elif rx:
        _, _, _, chip_nums, channels = zip(*[reverse_conversion(rx=idx) for idx in range(len(new_data))])
        pas = [""] * len(new_data)
        chip_type = "Rx"
    else:
        chip_nums = channels = pas = [""] * len(new_data)
        chip_type = "TxRx"


    df_new["DUT SN"] = ""
    df_new["Sys Type"] = ""
    df_new["Test Group"] = ""
    df_new["Test Name"] = test_name
    df_new["Board SN"] = ""
    df_new["Chip Type"] = chip_type
    df_new["Chip Num"] = chip_nums
    df_new["Channel"] = channels
    df_new["PA"] = pas
    df_new["Units"] = ""
    df_new["Min Limit ATE"] = ""
    df_new["Max Limit ATE"] = ""
    df_new["Verdict ATE"] = ""
    df_new["LOM Freq Config[MHz]"] = ""
    df_new["RF Freq Config[MHz]"] = ""
    df_new["BB Freq Config[MHz]"] = ""
    df_new["Error Msg"] = ""
    df_new["DTS[C]"] = ""
    df_new["Iteration"] = ""
    df_new["Chip ATE SN"] = ""
    df_new["OTP ID"] = ""
    df_new["OTP Version"] = ""
    df_new["Digital Backoff"] = ""

    df_new = df_new[list(template.keys())]
    df_combined = pd.concat([df_combined, df_new], ignore_index=True)

    print(f"Data added to DataFrame: {df_combined.tail(len(new_data))}")
    return df_combined
def filter_and_update_data(df_combined, global_verdict, test_name, min_limit, max_limit):
    """
    Filter the 'Result' values based on the provided limits and update the DataFrame with verdicts.

    Parameters:
        df_combined (pd.DataFrame): The DataFrame to filter and update.
        global_verdict (int): The initial verdict value, which will be updated based on the results.
        test_name (str): The name of the test to update.
        min_limit (float): The minimum acceptable limit for the 'Result' values.
        max_limit (float): The maximum acceptable limit for the 'Result' values.

    Returns:
        pd.DataFrame: The updated DataFrame.
        int: The updated global verdict value.
    """
    # Find rows where 'Test Name' matches
    mask = df_combined['Test Name'] == test_name
    df_filtered = df_combined[mask]

    # Check if there are any rows for the given test
    if not df_filtered.empty:
        # Update the 'Min Limit ATE' and 'Max Limit ATE' columns with the provided limits
        df_combined.loc[mask, 'Min Limit ATE'] = min_limit
        df_combined.loc[mask, 'Max Limit ATE'] = max_limit

        # Determine if the results are within limits and update the 'Verdict ATE' column
        def verdict(result):
            if pd.notna(result):
                if min_limit <= result <= max_limit:
                    return 1  # Passed
                else:
                    return 0  # Failed
            return None  # If result is NaN

        df_combined.loc[mask, 'Verdict ATE'] = df_combined.loc[mask, 'Result'].apply(verdict)

        # Update global verdict based on the results
        if (df_combined.loc[mask, 'Verdict ATE'] == 0).any():
            global_verdict = 0  # If any result is 0, set global verdict to 0
        else:
            global_verdict = 1  # Otherwise, set global verdict to 1

        print(f"Data updated for test '{test_name}':")
        print(df_combined[mask])
        print(f"Global Verdict: {global_verdict}")
    else:
        print(f"No data found for test '{test_name}'.")

    return df_combined, global_verdict


gmmt = [1.2, 3.4, 2.2]
gmmr = [1.5, 2.3, 4.5]
snr_ave = [1.5]
add_new_data('GMMt', gmmt, tx=True)
add_new_data('GMMr', gmmr, rx=True)
add_new_data("SNR_Average", snr_ave)

print(df_combined)
global_verdict = 1

# Example usage
df_combined, global_verdict = filter_and_update_data(df_combined, global_verdict, 'GMMt', 1.0, 3.0)  # Min and max limits for GMMt
print(global_verdict)
df_combined, global_verdict = filter_and_update_data(df_combined, global_verdict, 'GMMr', 1.0, 4.0)  # Min and max limits for GMMr
print(global_verdict)
df_combined, global_verdict = filter_and_update_data(df_combined, global_verdict, 'SNR_Average', 1.0, 4.0)  # Min and max limits for GMMr

print(df_combined)
print(global_verdict)
pass
