


dict_values_elements = {}
# combinations = {}
def power_per_element(chip, channel, pa, freq, value='', temp='' , save_value=False, get_value=False):

    if save_value:
        if chip not in dict_values_elements:
            dict_values_elements[chip] = {}

        if channel not in dict_values_elements[chip]:
            dict_values_elements[chip][channel] = {}

        if pa not in dict_values_elements[chip][channel]:
            dict_values_elements[chip][channel][pa] = {}

        dict_values_elements[chip][channel][pa][freq] = [value, temp]

    if get_value:
        if chip in dict_values_elements and channel in dict_values_elements[chip] and pa in dict_values_elements[chip][channel] and freq in dict_values_elements[chip][channel][pa]:
            return dict_values_elements[chip][channel][pa][freq]
        else:
            return None


chip_list = [0, 1, 2, 3, 4, 5]
chain_list = [0, 1, 2, 3]
Pa_el = [1, 2]
freq_list = [79.2, 77.7]
for freq_ghz in freq_list:
    for current_chip in chip_list:
        for current_chain in chain_list:
            for pa in Pa_el:
                print(current_chip, current_chain, pa, freq_ghz)
                # value = pa+current_chain+current_chip
                value = pa
                temp = 5


                power_per_element(current_chip, current_chain, pa, freq_ghz, value, temp, save_value=True)

                get_value = power_per_element(current_chip, current_chain, pa, freq_ghz ,get_value=True)
                z = get_value[int(value)]
                print(value)