# import sympy as sp
#
# # הגדרת המשתנים
# a, x, y, z = sp.symbols('a x y z')
#
# # הגדרת x כתלוי ב-a
# x = 2 + 9 * a
#
# # הגדרת הפונקציה
# f = x**3 + 2*y*z - sp.sin(x*z)
#
# # הנגזרת של הפונקציה לפי a
# df_da = sp.diff(f, a)
#
# print("הנגזרת של הפונקציה f לפי a:")
# print(df_da)
import numpy as np

x = [1, 0, 0]
y = [0, 1, 0]

if x == y:
    print("המערכים שווים")
else:
    print("המערכים לא שווים")