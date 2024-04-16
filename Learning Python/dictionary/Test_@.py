#
# y = []
# for x in range(5):
#     print(x)
#     y.append(x)
#
# print(y)
# verdict_Dcq = 0
# reason = "ok"
# verdict_Aqi = 0
# reason = "ok"
# verdict_Aqq = 1
# reason = "ok"
# reason = []
#
# if verdict_Dcq == 0 and verdict_Aqi == 0 and verdict_Aqq == 0:
#     print("0")
# else:
#     reason.append(reason)
#     print("1")

# z = "4"
# x = 'x'
# y = z + '_' + x
# print(y)
import numpy as np




# TI_ADC_DC_i = [1, 3, 5, 6, 8, 17, 12, 11]
# TI_ADC_DC_q = [0, 2, 4, 6, 8, 99, 0, 1]
# TI_ADC_inv_gain_i = [5, 3, 5, 9, 8, 17, 12, 11]
# TI_ADC_inv_gain_q = [5, 2, 4, 9, 8, 99, 0, 1]

# השתמש בפונקציה zip כדי לאחד את המערכים לזוגות וב-func אנומרי לקבלת האינדקסים והערכים
for idx in range(8):
    value_DC_i = TI_ADC_DC_i[idx]
    var_name_DC_i = f"DC_i_{idx}"
    print(var_name_DC_i)

    value_DC_q = TI_ADC_DC_q[idx]
    TI_ADC_DC_q = f"DC_q_{idx}"
    print(TI_ADC_DC_q)

    value_gain_i = TI_ADC_inv_gain_i[idx]
    TI_ADC_inv_gain_i = f"Gain_i_{idx}"
    print(TI_ADC_inv_gain_i)

    value_gain_q = TI_ADC_inv_gain_q[idx]
    TI_ADC_inv_gain_q = f"Gain_q_{idx}"
    print(TI_ADC_inv_gain_q)

# def rx_get_otp_date(chip):
#     address = 316
#     d1 = chip*2
#     d2 = chip*3
#     d3 = chip*4
#     d4 = chip*5
#     d5 = chip*6
#     d6 = chip*8
#
#     date_of_write = f'{d4}{d5}{d6}{d3}{d2}{d1}'
#     return date_of_write
#
#
# def rx_chips_otp():
#     chips = [0,1]
#     for chip in chips:
#         date_of_write = rx_get_otp_date(chip)
#     print(date_of_write)
#
# rx_chips_otp()

def rx_get_otp_date(chip):
    address = 316
    d1 = chip*2
    d2 = chip*3
    d3 = chip*4
    d4 = chip*5
    d5 = chip*6
    d6 = chip*8

    date_of_write = f'{d4}{d5}{d6}{d3}{d2}{d1}'
    return date_of_write


def rx_chips_otp():
    chips = [0, 1,2,3,4,5,6,7,8,9]
    date_of_write_dict = {}  # מילון ריק

    for chip in chips:
        date_of_write = rx_get_otp_date(chip)
        date_of_write_dict[chip] = date_of_write  # הוספת ערך למילון עם המפתח המתאים

    print(date_of_write_dict)


rx_chips_otp()