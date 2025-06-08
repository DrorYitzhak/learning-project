# Re-inject the full user code with English comments only, no code changes

def full_json_validation(system_type, json_file_path):
    """
        Perform validation on a calibration JSON file for BSRC or BSR32 systems.

        This function checks each calibration parameter in the JSON file and performs two types of validation:
        1. **Zero-value index validation**: Certain predefined indices (per parameter) are expected to have a value of 0.
           These are excluded from range validation but explicitly checked to be exactly zero.
        2. **Range validation**: All other indices (not excluded) are validated to be within an allowed numerical range
           defined per parameter.

        Parameters:
        ----------
        system_type : str
            Either 'bsrc' or 'bsr32'. Determines which set of rules to apply.
        json_file_path : str
            Full path to the calibration JSON file.

        Returns:
        -------
        full_report : list of dict
            A structured report of validation results per parameter.
            Each item includes:
                - test: The parameter name
                - status: 'PASSED' or 'FAILED'
                - excluded_index_errors: List of (index, value) where 0 was expected but not found
                - range_errors: List of (index, value) that fell outside the valid range
                - total_checked: Count of validated (non-excluded) indices
                - total_failed: Total number of errors for this parameter
        """
    import json

    # Load the JSON calibration file
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # Predefined indices that should contain zero values for bsr32
    excluded_indices_bsr32 = {
        'd_rxbb_bw700_q2_2_i': [],
        'd_rxbb_bw700_q2_1_i': [],
        'd_rxbb_bw700_wo2_1_i': [],
        'd_rxbb_bw700_wo1_i': [],
        'rx_dig_lo_amp_bg_const_curr_tune': [0, 1, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 21, 22, 23, 24,
                                             25, 27, 28, 29, 30, 31, 33, 34, 35, 36, 37, 39, 40, 41, 42, 43, 45, 46, 47],
        'rx_dig_bg_const_curr_tune': [],
        'dig_rxbb_700_filt_ctrl': [],
        'dig_rx_plpf_cbank_0v8': [],
        'tx_dig_lo_amp_bg_const_curr_tune': [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
        'tx_dig_bg_const_curr_tune': [16, 17, 18, 19, 20, 21, 22, 23],
        'tx_aqi': [16, 17, 18, 19, 20, 21, 22, 23],
        'tx_aqq': [16, 17, 18, 19, 20, 21, 22, 23],
        'rx_aqi': [],
        'rx_aqq': [],
        'tx_dc_i_a': [16, 17, 18, 19, 20, 21, 22, 23],
        'tx_dc_q_a': [16, 17, 18, 19, 20, 21, 22, 23],
        'tx_dc_i_b': [16, 17, 18, 19, 20, 21, 22, 23],
        'tx_dc_q_b': [16, 17, 18, 19, 20, 21, 22, 23],
    }

    # Allowed value ranges for bsr32
    validation_ranges_bsr32 = {
        'tx_aqi': (-500, 500),
        'tx_aqq': (1500, 2500),
        'rx_dig_lo_amp_bg_const_curr_tune': (5, 30),
        'tx_dig_lo_amp_bg_const_curr_tune': (5, 30),
        'tx_dig_bg_const_curr_tune': (5, 30),
        'tx_dc_i_a': (-800, 1000),
        'tx_dc_q_a': (-800, 1000),
        'tx_dc_i_b': (-800, 1000),
        'tx_dc_q_b': (-800, 1000),
        'd_rxbb_bw700_q2_1_i': (3, 8),
        'd_rxbb_bw700_q2_2_i': (3, 8),
        'd_rxbb_bw700_wo1_i': (3, 8),
        'd_rxbb_bw700_wo2_1_i': (3, 10),
        'dig_rx_plpf_cbank_0v8': (15, 30),
        'dig_rxbb_700_filt_ctrl': (5, 15),
        'rx_aqi': (-500, 500),
        'rx_aqq': (1500, 25000),
        'rx_dig_bg_const_curr_tune': (5, 30),
    }

    # Predefined indices for bsrc extracted from actual BSRC JSON
    excluded_indices_bsrc = {
        'd_rxbb_bw700_q2_2_i': [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47],
        'd_rxbb_bw700_q2_1_i': [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47],
        'd_rxbb_bw700_wo2_1_i': [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47],
        'd_rxbb_bw700_wo1_i': [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47],
        'rx_dig_lo_amp_bg_const_curr_tune': [0, 1, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26,
                                             27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47],
        'rx_dig_bg_const_curr_tune': [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47],
        'dig_rxbb_700_filt_ctrl': [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47],
        'dig_rx_plpf_cbank_0v8': [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47],
        'tx_dig_lo_amp_bg_const_curr_tune': [0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
        'tx_dig_bg_const_curr_tune': [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
        'tx_aqi': [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
        'tx_aqq': [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
        'rx_aqi': [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47],
        'rx_aqq': [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47],
        'tx_dc_i_a': [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
        'tx_dc_q_a': [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
        'tx_dc_i_b': [6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
        'tx_dc_q_b': [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
    }

    # Allowed value ranges for bsrc
    validation_ranges_bsrc = validation_ranges_bsr32

    # Choose the relevant rules based on the system type
    if system_type == 'bsr32':
        excluded_indices = excluded_indices_bsr32
        validation_ranges = validation_ranges_bsr32
    elif system_type == 'bsrc':
        excluded_indices = excluded_indices_bsrc
        validation_ranges = validation_ranges_bsrc
    else:
        raise ValueError("Invalid system_type. Must be 'bsr32' or 'bsrc'.")

    full_report = []

    # Validate each parameter
    for param, indices_to_exclude in excluded_indices.items():
        values = data.get(param, [])
        excluded_errors = []
        included_values = []

        for i, v in enumerate(values):
            if i in indices_to_exclude:
                if v != 0:
                    excluded_errors.append((i, v))  # Should be 0 but it's not
            else:
                included_values.append((i, v))    # Should be in range

        range_errors = []
        min_val, max_val = validation_ranges[param]
        for i, v in included_values:
            if not (min_val <= v <= max_val):
                range_errors.append((i, v))       # Out of range

        if excluded_errors or range_errors:
            full_report.append({
                "test": param,
                "status": "FAILED",
                "excluded_index_errors": [{"index": i, "value": v} for i, v in excluded_errors],
                "range_errors": [{"index": i, "value": v} for i, v in range_errors],
                "total_checked": len(included_values),
                "total_failed": len(excluded_errors) + len(range_errors)
            })
        else:
            full_report.append({
                "test": param,
                "status": "PASSED",
                "total_checked": len(included_values),
                "total_failed": 0
            })

    return full_report




report = full_json_validation(
    system_type='bsr32',
    json_file_path=r'C:\Users\drory\Downloads\cal_32_76128_prod_mrr_Jacob_Dror_debug_1.json'
)

import json
print(json.dumps(report, indent=2))

