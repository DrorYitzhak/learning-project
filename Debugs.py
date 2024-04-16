import requests

def fetch_html(url):
    try:
        response = requests.get(url)
        # בדיקת תגובת השרת
        if response.status_code == 200:
            return response.text
        else:
            print("Failed to fetch HTML. Status code:", response.status_code)
            return None
    except Exception as e:
        print("An error occurred:", e)
        return None

# הפעלת הפונקציה עם כתובת ה-URL של האתר שברצונך לקבל את ה-HTML שלו
html_content = fetch_html("https://www.yad2.co.il/realestate/item/cm5ng7tb?ad-location=Main+feed+listings&opened-from=Feed+view&component-type=main_feed&index=1&color-type=Platinum")

# הדפסת קוד ה-HTML אם הוא נשלף בהצלחה
if html_content:
    print(html_content)
