def json_results_array(key, value, my_dict=[]):
    my_dict.append({key: value})
    return my_dict

# דוגמה לשימוש:
my_dictionary = json_results_array("name", "John")
print(my_dictionary)

json_results_array("age", 25, my_dictionary)
print(my_dictionary)

json_results_array("age2", 25, my_dictionary)
print(my_dictionary)

json_results_array("age2", 25, my_dictionary)
print(my_dictionary)