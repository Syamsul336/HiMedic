from flask import Flask, render_template, request
import pandas as pd
import math
from sklearn.metrics.pairwise import cosine_similarity
import re

app = Flask(__name__)



# memanggil data gejala-gejala penyakit 
data_symptoms = pd.read_csv("dataset/desease_symptoms.csv",sep=";")

# memanggil data yang berisi penjelasan singkat, gaya hidup yang disarankan dan makanan yang disarankan tentang suatu penyakit
data_food_style = pd.read_csv("dataset/food_n_style.csv",sep=";")

# membuat list yang merupakan kumpulan dari semua gejala secara keseluruhan
gejala_all = list(set(data_symptoms["gejala"]))

# membuat daftar semua nama penyakit yang ada dalam data
penyakit_label = data_symptoms["penyakit"].unique()

# Membuat Dataframe kosong yang akan akan berisi nilai kategori hubungan kolesterol, asam urat, gula darah (tiga variabel)
data_autocheck = pd.DataFrame(columns = ["kolesterol", "asam urat", "gula darah"])

# Perulangan tiap baris pada data penyakit diatas 
for penyakit in penyakit_label:

  # membuat list gejala masing penyakit dalam setiap perulangan
  gejala_temp = list(data_symptoms.loc[data_symptoms["penyakit"] == penyakit]["gejala"])
  temp = [0,0,0] # vector sementara yang nilainya akan diubah lalu nantinya ditambahkan menjadi baris baru pada datframe data_autocheck

  # perulangan untuk memeriksa hubungan tiap penyakit dengan tiga variabel diatas
  # 2 berarti tinggi
  # 1 berarti rendah
  # 0 berarti tidak ada hubungan
  # setiap nilai yang ditentukan akan ditambahkan ke vector temp
  for gejala in gejala_temp:

    # mengecek hubungan penyakit dengan kolesterol
    if "kolesterol (tinggi)" in gejala_temp:
      temp[0] = 2
    elif "kolesterol (rendah)" in gejala_temp:
      temp[0] = 1
    elif "kolesterol (tidak ada hubungan)" in gejala_temp:
      temp[0] = 0
    else:
      pass
    
    # mengecek hubungan penyakit dengan asam urat
    if "asam urat (tinggi)" in gejala_temp:
      temp[1] = 2
    elif "asam urat (rendah)" in gejala_temp:
      temp[1] = 1
    elif "asam urat (tidak ada hubungan)" in gejala_temp:
      temp[1] = 0
    else:
      pass
    
    # mengecek hubungan penyakit dengan gula darah
    if "gula darah (tinggi)" in gejala_temp:
      temp[2] = 2
    elif "gula darah (rendah)" in gejala_temp:
      temp[2] = 1
    elif "gula darah (tidak ada hubungan)" in gejala_temp:
      temp[2] = 0
    else:
      pass
  
  # menambahkan vector temp ke datframe data_autocheck
  data_autocheck.loc[len(data_autocheck)] = temp

# karena kita telah mentransformasi tiga variabel awal ke dalam bentuk baru
# maka kita akan menghapus tiga variabel dari dataset data_symptoms

list_autocheck_del = ["kolesterol (tinggi)", "kolesterol (rendah)", "kolesterol (tidak ada hubungan)",
                      "asam urat (tinggi)", "asam urat (rendah)", "asam urat (tidak ada hubungan)",
                      "gula darah (tinggi)","gula darah (rendah)","gula darah (tidak ada hubungan)"]

for word in list_autocheck_del:
  data_symptoms = data_symptoms[data_symptoms.gejala != word]
data_symptoms = data_symptoms.reset_index(drop=True)

# membuat dataframe data_enc yang akan berisi data gejala penyakit dalam bentuk vector
data_enc = pd.DataFrame(columns = gejala_all)

# perulangan pada setiap penyakit 
for penyakit in penyakit_label:
  gejala_temp = list(data_symptoms.loc[data_symptoms["penyakit"] == penyakit]["gejala"])
  temp = [] # menampung sementara vector berisi nilai encoding setiap penyakit
  
  # perulangan pada gejala setiap penyakit
  for gejala in gejala_all:
    if gejala in gejala_temp:
      temp.append(1) # jika penyakit memiliki gejala tersebut maka kolom akan menunjukkan 1
    else:
      temp.append(0) # jika penyakit tidak memiliki gejala tersebut maka kolom akan menunjukkan 0
  data_enc.loc[len(data_enc)] = temp

name = list(data_enc.sum(axis = 0, skipna = True).sort_values(ascending=False).index)

data_enc = data_enc[name]

data_enc = pd.concat([data_autocheck, data_enc.reindex(data_autocheck.index)], axis=1)


flag_counter = 0
list_gejala_user = []
data_enc2 = data_enc
kolesterol_user = 0
asam_urat_user = 0
gula_darah_user = 0 
input_gejala = 0
hasil_diagnosa = ""
persentase = ""
penjelasan_penyakit = ""
gaya_hidup_penyakit = ""
makanan_penyakit = ""
reset_counter = 0



def cosim_diagnosis():
  global reset_counter, flag_counter, list_gejala_user, data_enc2, kolesterol_user, asam_urat_user, gula_darah_user, data_enc, input_gejala, hasil_diagnosa, persentase, penjelasan_penyakit, gaya_hidup_penyakit, makanan_penyakit
  if flag_counter == "stop":
    print("udah bos")
    return
  if flag_counter == 0:
    # meminta inputan awal berupa data kolesterol, asam urat, gula darah
    # kolesterol_user = int(input(f"masukkan kadar kolesterol : "))
    # asam_urat_user = int(input(f"masukkan kadar asam urat : "))
    # gula_darah_user = int(input(f"masukkan kadar gula darah : "))

    # mengkonversi data jumlah kedalam data kategorik karena dalam dataframe kolesterol, asam urat dan gula darah memiliki nilai kategorik (0,1,2)
    kolesterol_user = 1 if kolesterol_user < 240 else 2
    asam_urat_user = 1 if asam_urat_user < 6.5 else 2
    gula_darah_user = 1 if gula_darah_user < 180 else 2

    # list gejala user merupakan vector inputan yang akan dihitung bersamaan dengan datframe data_enc2 menggunakan cosine similarity
    # telah terdapat 3 nilai awal yang merupakan inputan pada baris sebelumnya
    list_gejala_user = [kolesterol_user, asam_urat_user, gula_darah_user]
    flag_counter += 3
  elif flag_counter > 0:
    total = sum(data_enc2.iloc[:,len(list_gejala_user)])
    total_index =  len(data_enc2.index)
    if total == total_index:
      list_gejala_user.append(1)
      flag_counter += 1
      cosim_diagnosis()
      return
    elif total == 0:
      list_gejala_user.append(0)
      flag_counter += 1
      cosim_diagnosis()
      return
    else:
      # input_gejala = input(f"apakah anda merasa {data_enc2.columns[flag_counter]} ")
      list_gejala_user.append(int(input_gejala))   
      flag_counter += 1

    len_input = len(list_gejala_user)
    if len_input%2 == 0:
      print("yokai")
      data_temp = data_enc2.iloc[:,:len_input]


      data_temp = data_temp.append(pd.DataFrame([list_gejala_user], columns=list(data_temp.columns)), ignore_index=False)


      similarity = pd.DataFrame(cosine_similarity(data_temp))


      similarity = similarity.iloc[0:len(similarity)-1]

      similarity = similarity.set_index(pd.Series(list(data_enc2.index)))

      similarity = similarity.sort_values(by=len(similarity), ascending=False)

      index_similarity = similarity.iloc[0:math.ceil(0.6*len(similarity))].index

      data_enc2 = data_enc2.loc[list(index_similarity)]

    # jika list penyakit tersisa 3 saja, maka perulangan berhenti
    
    hasil_diagnosa = penyakit_label[list(data_enc2.index)]
    hasil_diagnosa_awal = hasil_diagnosa
    order = r'[0123456789]'
    temp_diagnosa = []
    for j in hasil_diagnosa:
      filtered_string = re.sub(order, '', j)
      temp_diagnosa.append(filtered_string)

    hasil_diagnosa = temp_diagnosa

    if len(hasil_diagnosa) <= 3:

      persentase = list(similarity.iloc[:,-1].loc[list(data_enc2.index)])
      penjelasan_penyakit = list(data_food_style.loc[list(data_enc2.index), "Penjelasan Singkat"])
      gaya_hidup_penyakit = list(data_food_style.loc[list(data_enc2.index), "Saran Gaya Hidup"])
      makanan_penyakit = list(data_food_style.loc[list(data_enc2.index), "Saran Makanan"])


      # dictionary of lists 
      dict = {'hasil_diagnosa': hasil_diagnosa, 'persentase': persentase, 'penjelasan_penyakit': penjelasan_penyakit,
              'gaya_hidup_penyakit' : gaya_hidup_penyakit, 'makanan_penyakit' : makanan_penyakit} 
          
      df_hasil = pd.DataFrame(dict)
      df_hasil = df_hasil.drop_duplicates(subset='hasil_diagnosa', keep="first")

      hasil_diagnosa = df_hasil["hasil_diagnosa"].values.tolist()
      persentase = df_hasil["persentase"].values.tolist()
      penjelasan_penyakit = df_hasil["penjelasan_penyakit"].values.tolist()
      gaya_hidup_penyakit = df_hasil["gaya_hidup_penyakit"].values.tolist()
      makanan_penyakit = df_hasil["makanan_penyakit"].values.tolist()

      #print(hasil_diagnosa)
      #print(persentase)
      #print(penjelasan_penyakit)
      #print(gaya_hidup_penyakit)
      #print(makanan_penyakit)
      
      flag_counter = -1


@app.route('/')
def home():
    global flag_counter
    return render_template('index.html', 
                            flag_counter = flag_counter, 
                            flag_counter2=flag_counter,
                            progress = (1 - len(data_enc2.index)/len(data_enc.index))*100)



deteksi = 0
@app.route("/predict", methods=["POST"])
def predict():
    global flag_counter, list_gejala_user, data_enc2, kolesterol_user, asam_urat_user, gula_darah_user, data_enc, input_gejala, hasil_diagnosa, persentase, penjelasan_penyakit, gaya_hidup_penyakit, makanan_penyakit, reset_counter
    if flag_counter == 0:
        kolesterol_user = int(request.form['kadar_kolseterol'])
        asam_urat_user = int(request.form['kadar_asam_urat'])
        gula_darah_user = int(request.form['kadar_gula_darah'])
        cosim_diagnosis()
        return render_template('index.html', 
                                pertanyaan_gejala=data_enc.columns[flag_counter], 
                                flag_counter = flag_counter, 
                                flag_counter2=flag_counter, 
                                flag_counter3 = flag_counter,
                                progress = (1 - len(data_enc2.index)/len(data_enc.index))*100 )
    elif flag_counter >= 3:
        input_gejala = int(request.form['is_menderita_gejala'])
        cosim_diagnosis()
        if flag_counter == -1:
            if len(hasil_diagnosa) == 2:
                reset_counter = 1
                return render_template('index.html',  
                                    flag_counter = flag_counter, flag_counter2 = flag_counter, flag_counter3 = flag_counter, 
                                    flag_counter4 = flag_counter, reset_counter = reset_counter,
                                    hasil_diagnosa1 = hasil_diagnosa[0], hasil_diagnosa2= hasil_diagnosa[1],
                                    persentase1 = int(persentase[0]*100), persentase2 = int(persentase[1]*100),
                                    penjelasan1 = penjelasan_penyakit[0], penjelasan2 = penjelasan_penyakit[1],
                                    gaya_hidup1 = gaya_hidup_penyakit[0], gaya_hidup2 = gaya_hidup_penyakit[1],
                                    makanan_penyakit1 = makanan_penyakit[0], makanan_penyakit2 = makanan_penyakit[1], len_output = len(hasil_diagnosa))
            else :
                reset_counter = 1
                return render_template('index.html',  
                                    flag_counter = flag_counter, flag_counter2 = flag_counter, flag_counter3 = flag_counter, 
                                    flag_counter4 = flag_counter, reset_counter = reset_counter, 
                                    hasil_diagnosa1 = hasil_diagnosa[0], hasil_diagnosa2= hasil_diagnosa[1], hasil_diagnosa3= hasil_diagnosa[2], 
                                    persentase1 = int(persentase[0]*100), persentase2 = int(persentase[1]*100), persentase3 = int(persentase[2]*100),
                                    penjelasan1 = penjelasan_penyakit[0], penjelasan2 = penjelasan_penyakit[1], penjelasan3 = penjelasan_penyakit[2],
                                    gaya_hidup1 = gaya_hidup_penyakit[0], gaya_hidup2 = gaya_hidup_penyakit[1], gaya_hidup3 = gaya_hidup_penyakit[2],
                                    makanan_penyakit1 = makanan_penyakit[0], makanan_penyakit2 = makanan_penyakit[1],  makanan_penyakit3 = makanan_penyakit[2], len_output = len(hasil_diagnosa))               

        else :
            return render_template('index.html', 
                                    pertanyaan_gejala=data_enc.columns[flag_counter], 
                                    flag_counter = flag_counter, 
                                    flag_counter2=flag_counter, 
                                    flag_counter3 = flag_counter,
                                    progress = (1 - len(data_enc2.index)/len(data_enc.index))*100 )
    if reset_counter == 1:
        reset_counter = 0
        flag_counter = 0
        list_gejala_user = []
        data_enc2 = data_enc
        kolesterol_user = 0
        asam_urat_user = 0
        gula_darah_user = 0 
        input_gejala = 0
        hasil_diagnosa = ""
        persentase = ""
        penjelasan_penyakit = ""
        gaya_hidup_penyakit = ""
        makanan_penyakit = ""
        return render_template('index.html', 
                            flag_counter = flag_counter, 
                            flag_counter2=flag_counter,
                            progress = (1 - len(data_enc2.index)/len(data_enc.index))*100)


    

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)