import serial
import time
import numpy as np
import matplotlib.pyplot as plt

# Apri la connessione seriale (sostituisci '/dev/ttyACM0' con la tua porta seriale)
ser = serial.Serial('COM14', 250000)

# Aspetta che il microcontrollore mandi modalità corrente e data rate

# Invia il comando di start al microcontrollore
time.sleep(1)
ser.write('s'.encode())
time.sleep(1)

# Aspetta che il microcontrollore mandi modalità corrente e data rate
while True:
    if ser.inWaiting():
        str = ser.readline().decode().strip()
        print("Stringa ricevuta: " + str)
        break


# Inizializza la matrice e l'array per i dati
data_matrix = []
data_array = []

try:
    while True:
        if ser.inWaiting():
            start_byte = ser.read(1)  # Leggi il byte di inizio
            if start_byte == b'\xCC':  # Verifica il byte di inizio
                high_byte = ser.read(1)  # Leggi l'high byte
                low_byte = ser.read(1)  # Leggi il low byte
                measurement = (ord(high_byte) << 8) | ord(low_byte)  # Unisci i byte
                data_array.append(measurement)  # Aggiungi la misurazione all'array
                if len(data_array) == 860:  # Se l'array ha raggiunto la lunghezza desiderata
                    data_matrix.append(data_array)  # Aggiungi l'array alla matrice
                    data_array = []  # Resetta l'array
except KeyboardInterrupt:
    # Quando il programma viene interrotto, salva la matrice in un file di testo
    data_matrix = data_matrix[1:]  # Rimuovi la prima riga (incompleta)
    np.savetxt('data_matrix.txt', data_matrix)
    print("Dati salvati in 'data_matrix.txt'")
finally:
    ser.close()  # Chiudi la connessione seriale

    # Appiattisci la matrice in un array unidimensionale
    data_array = np.concatenate(data_matrix)

    # Crea un array di tempi. Ogni 860 campioni corrispondono a un secondo.
    time_array = np.arange(len(data_array)) / 860

    # Crea un grafico
    plt.figure(figsize=(10, 6))
    plt.plot(time_array, data_array)
    plt.title('Grafico dei dati')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Valore')

    # Mostra il grafico
    plt.show()