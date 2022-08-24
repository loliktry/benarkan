from firebase import firebase
firebase = firebase.FirebaseApplication('https://berita-benar-default-rtdb.asia-southeast1.firebasedatabase.app/', None)
result = firebase.get('/Berita', None)
print (result)