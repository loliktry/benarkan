from firebase import firebase
firebase = firebase.FirebaseApplication('https://berita-benar-default-rtdb.asia-southeast1.firebasedatabase.app/', None)

#result = firebase.post('/Berita', {'Judul':'Permintaan Maaf Bima Sakti Usai Timnas U16 Indonesia Lolos ke Final Piala AFF U16 2022','Konten':'Timnas U16 Indonesia melangkah ke final seusai menumbangkan Myanmar 5-4 (1-1) via adu penalti, Rabu (10/8/2022) malam WIB. Pertandingan timnas U16 Indonesia vs Myanmar yang dihelat di Stadion Maguwoharjo berlangsung sangat sengit. Timnas U16 Indonesia harus tertinggal 0-1 terlebih dahulu setelah gelandang Myanmar, Nay Min Htet, mencetak gol pada menit ke-43. Usaha timnas U16 Indonesia untuk menyamakan kedudukan kemudian membuahkan hasil pada menit ke-70 berkat gol tendangan bebas M. Riski Afrisal.','Judul':'Irjen Ferdy Sambo Terima Gaji Besar, Berapa Total Per Bulan?','Konten':'Ferdy Sambo pun kini harus berakhir gegara disangkakan menjadi otak pelaku pembunuhan tersebut. Irjen Ferdy Sambo terakhir menjabat Kadiv Propam Polri. Jabatan itu pun kini harus ditanggalkannya setelah Kapolri memutasi Irjen Ferdy Sambo sebagai pati Yanma Polri. Kapolri Jenderal Listyo Sigit Prabowo sendiri yang mencopot jabatan Irjen Ferdy Sambo melalui TR 1628/VIII/KEP/2022/4 Agustus 2022. Sebagai Pati Polri, Ferdy Sambo tentunya mendapatkan berbagai fasilitas negara untuk menunjang tugasnya. Sebut saja rumah dinas di kawasan elit, Duren Tiga, yang belakangan jadi tempat terbunuhnya Brigadir J.'})
result = firebase.post('/Berita', {'Judul':'Irjen Ferdy Sambo Terima Gaji Besar, Berapa Total Per Bulan?','Konten':'Ferdy Sambo pun kini harus berakhir gegara disangkakan menjadi otak pelaku pembunuhan tersebut. Irjen Ferdy Sambo terakhir menjabat Kadiv Propam Polri. Jabatan itu pun kini harus ditanggalkannya setelah Kapolri memutasi Irjen Ferdy Sambo sebagai pati Yanma Polri. Kapolri Jenderal Listyo Sigit Prabowo sendiri yang mencopot jabatan Irjen Ferdy Sambo melalui TR 1628/VIII/KEP/2022/4 Agustus 2022. Sebagai Pati Polri, Ferdy Sambo tentunya mendapatkan berbagai fasilitas negara untuk menunjang tugasnya. Sebut saja rumah dinas di kawasan elit, Duren Tiga, yang belakangan jadi tempat terbunuhnya Brigadir J.'})
print (result)