from firebase import firebase
db = firebase.FirebaseApplication("https://berita-benar-default-rtdb.asia-southeast1.firebasedatabase.app/", None)

data = {
    'nama':'datang',
    'email':'pergi',
    'no':20
}

result = firebase.post("https://berita-benar-default-rtdb.asia-southeast1.firebasedatabase.app/", data)
print(result)