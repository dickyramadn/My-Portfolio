import numpy as np

def topsis(data, bobot, impact, alternatif):
    # hitung normalized matrix
    pembagi = []
    for i in range(len(data[0])):
        dataC = []
        for j in range(len(data)):
            dataC.append(data[j][i])
        pembagi.append(np.linalg.norm(dataC))
    normalized_matrix = data/pembagi
    # cek hasil
    #print(f'R = {normalized_matrix}')
    
    # hitung weight normalized matrix
    weight_normalized_matrix = normalized_matrix * bobot
    # cek hasil
    #print(f'V = {weight_normalized_matrix}' )
    
    # hitung solusi ideal positif dan negatif
    ideal_positif = []
    ideal_negatif = []
    WNM = []
    for i in range(len(data[0])):
        wnm = []
        for j in range(len(data)):
            wnm.append(weight_normalized_matrix[j][i])
        WNM.append(wnm)
    for i in range(len(WNM)):
        if impact[i] == 'cost':
            ideal_positif.append(min(WNM[i]))
            ideal_negatif.append(max(WNM[i]))
        else :
            ideal_positif.append(max(WNM[i]))
            ideal_negatif.append(min(WNM[i]))
            
    # cek hasil
    print(f'MAX = {ideal_positif}')
    print(f'MIN = {ideal_negatif}')
    
    # hitung jarak dari solusi ideal positif dan negatif
    jarak = []
    for i in range(len(data)):
        dp = np.sqrt(np.sum((ideal_positif - weight_normalized_matrix[i])**2))
        dn = np.sqrt(np.sum((ideal_negatif - weight_normalized_matrix[i])**2))
        jarak.append(dn/(dp+dn))
        # cek hasil
        #print(f'D{i+1}+ = {dp}')
        #print(f'D{i+1}- = {dn}')
    
    # urutkan hasil
    for i in range(len(jarak)):
        peringkat = 1
        for j in range(len(jarak)):
            if jarak[i] < jarak[j] :
                peringkat += 1
        print( f'{alternatif[i]} peringkat {peringkat} dengan nilai prferensi {jarak[i]}' )
    
    return jarak

# masukan data
data = []
bobot = []
impact = []
alternatif = []
kriteria = []

jumlahKri = int(input('Jumlah kriteria yang akan digunakan : '))
print('')
for i in range(jumlahKri) :
    Kri = input(f'Masukan kriteria ke-{i+1} : ')
    kriteria.append(Kri)
print('')
for i in range(jumlahKri) :
    bbt = float(input(f'Masukan bobot untuk {kriteria[i]} : '))
    bobot.append(bbt)
    im = input('Jenis kriteria (benefit/cost) : ').lower()
    impact.append(im)
print('-'*70)
jumlahAlt = int(input('Jumlah alternatif yang akan diproses : '))
for i in range(jumlahAlt) :
    Alt = input(f'Masukan alternatif ke-{i+1} : ')
    alternatif.append(Alt)
print('')
print(f'Masukan data alternatif untuk setiap kriteria.\nTolong masukan dalam urutan {kriteria} !' )
for i in range(jumlahAlt) :
    dt = input(f'{alternatif[i]} : ')
    dt = dt.split(",")
    data.append([float(x) for x in dt])

data = np.array(data)
bobot = np.array(bobot)
print('-'*70)

# hitung pilihan terbaik menggunakan metode topsis
hasil = topsis(data, bobot, impact, alternatif)
