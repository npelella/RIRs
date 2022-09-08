from tkinter import *
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import numpy as np
import scipy.io.wavfile 
from scipy import signal, ndimage, misc
from matplotlib import pyplot as plt
import soundfile as sf
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy import stats
import math
import pandas as pd

# Raiz

raiz = Tk()
raiz.title("UNTreF")
raiz.iconbitmap("Untref.ico") 
#raiz.geometry("1100x700")
#raiz.config(bg="#31414b")
raiz.config(bg="#18232d")

# Frame

miFrame = Frame()
miFrame.pack(fill="both", expand="True")
miFrame.config(bg="#18232d")

#%% Functions

#----- Open File 

def abrirFichero():
    
    global fichero
    
    varBotonArchivo.set(0)
    varBotonFiltro.set(0)
    varBotonSuavizado_MM.set(0)
    varBotonSuavizado_Sch.set(0)
    varBoton_Channels.set(0)
    
    fichero = filedialog.askopenfilename(title="Abrir",filetypes=(("Ficheros de audio","*.wav")
    ,("Todos los ficheros","*.*")))
    
    print(fichero)
    

    
#----- File Reading
  

varBoton_Channels = IntVar()
varBotonArchivo = IntVar()
 
def lecturaArchivos():
    # This function reads the audio file according to the type of file (Mono/Stereo)
    
    global audio
    global audioEstereo
    global audioL
    global audioR
    global fs
    global t
    global T
    
    
    if varBoton_Channels.get() == 3:
        audioEstereo,fs = sf.read(fichero) # Audio File Reading
        audio = audioEstereo
        audioL = audioEstereo
        audioR = audioEstereo
        t = np.arange(len(audio))/fs       # Time (seconds) vector of the file
        T = t[len(t)-1]                    # Time (seconds) value
    elif varBoton_Channels.get()==1:
        audioEstereo,fs = sf.read(fichero) 
        audio = audioEstereo[:,0]
        audioL = audio
        audioR = audioEstereo[:,1]
        t = np.arange(len(audio))/fs 
        T = t[len(t)-1]
    elif varBoton_Channels.get()==2:
        audioEstereo,fs = sf.read(fichero) 
        audio = audioEstereo[:,1]
        audioL = audioEstereo[:,0]
        audioR = audio
        t = np.arange(len(audio))/fs 
        T = t[len(t)-1]
    else:
        return

lecturaArchivos()

#----- RIR obteining

def obtencionIR():
    # When the option is Sine Sweep, this function generates the inverse filter in order to obtain
    # the Room Impulse Response
    
    global IR, t_IR, IR_L, IR_R, t_IRL, t_IRR
    
    if varBotonArchivo.get()==1:

        fMin = int(cuadroFrecuencia_min.get())
        fMax = int(cuadroFrecuencia_max.get())
        T = int(cuadroDuracion.get())
        
        tAudio = np.arange(len(audio))/fs
        t = np.arange(0,T+1,1/fs)
        w1 = 2*np.pi*fMin              # Minimum angular frequency
        w2 = 2*np.pi*fMax              # Maximum angular frequency
        K = (w1/np.log(w2/w1))*T       
        L = T/np.log(w2/w1)
        y = np.sin(K*((np.exp(t/L))-1)) 
        
        # Modulation
        
        w = (K/L)*np.exp(t/L) 
        m = w1/(2*np.pi*w)
        
        # Inverse filter
        
        tInvertido = np.flip(t)
        yInvertido = np.sin(K*((np.exp(tInvertido/L))-1))
        k = m*yInvertido
        k = k/max(abs(k))
        k = np.concatenate((k,np.zeros(len(audio)-len(k))))
        kEspectro = np.fft.fft(k) 
         
        # RIR
        
        if varBoton_Channels.get() == 3:   
            audioEspectro = np.fft.fft(audio)
            IR = audioEspectro*kEspectro
            IR =np.fft.ifft(IR)
            IR=abs(IR[IR.argmax()-10:])
            t_IR = np.arange(len(IR))/fs
        else:
            audioEspectroL = np.fft.fft(audioL)
            audioEspectroR = np.fft.fft(audioR)
            IR_L = audioEspectroL*kEspectro
            IR_R = audioEspectroR*kEspectro
            IR_L =np.fft.ifft(IR_L)
            IR_R =np.fft.ifft(IR_R)
            IR_L=abs(IR_L[IR_L.argmax()-10:])
            IR_R=abs(IR_R[IR_R.argmax()-10:])
            t_IRL = np.arange(len(IR_L))/fs
            t_IRR = np.arange(len(IR_R))/fs
    else:
        return 
    
#----- Filtered

varBotonSuavizado_MM = IntVar()
varBotonSuavizado_Sch = IntVar()
varBotonFiltro = IntVar()
seleccion = StringVar()
seleccion2 = StringVar()

def filtrado():
    # This function filters the Impulse response by octave band or third octave band
    
    global audioFiltrado, frecuenciaCentral, desplegable, desplegable2, audio, audioL, audioR, audioFiltradoL, audioFiltradoR
    
    varBotonSuavizado_MM.set(0)
    varBotonSuavizado_Sch.set(0)
    
    if varBotonArchivo.get()==1:
        if varBoton_Channels.get()==3:
            obtencionIR()
            RIR_final=IR
        else:
            obtencionIR()
            RIR_finalL=IR_L
            RIR_finalR=IR_R
    elif varBotonArchivo.get()==2:
        if varBoton_Channels.get()==3:
            audio=audio[audio.argmax()-10:]
            t = np.arange(len(audio))/fs 
            T = t[len(t)-1]
            RIR_final=audio
        else:
            audioL=audioL[audioL.argmax():]
            tL = np.arange(len(audioL))/fs 
            TL = tL[len(tL)-1]
            RIR_finalL=audioL
            audioR=audioR[audioR.argmax():]
            tR = np.arange(len(audioR))/fs 
            TR = tR[len(tR)-1]
            RIR_finalR=audioR            
    else:
        return
        
    # Third band filtered
    
    if varBotonFiltro.get()==1:
        frecuenciaCentral = np.array([25, 31.5, 40, 50, 63, 80, 100, 125, 160,
    200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 
    6300, 8000, 10000, 12500, 16000])
        desplegable = ttk.Combobox(miFrame, textvariable=seleccion)
        desplegable["values"] = ('25', '31.5', '40', '50', '63', '80', '100', '125', '160', '200', '250', '315',
              '400', '500', '630', '800', '1k', '1.25k', '1.6k', '2k', '2.5k', '3.15k', '4k',
              '5k', '6.3k', '8k', '10k', '12.5k', '16k')
        desplegable.grid(row=1,column=4,padx=2,pady=2)
        
        desplegable2 = ttk.Combobox(miFrame, textvariable=seleccion2)
        desplegable2["values"] = ('25 to 200 Hz', '250 to 2k Hz', '2k to 16k Hz')
        desplegable2.grid(row=8,column=4,padx=2,pady=2)
    
    # Octave band filtered   
        
    elif varBotonFiltro.get() == 2:
        frecuenciaCentral = np.array([31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000])
        desplegable = ttk.Combobox(miFrame, textvariable=seleccion)
        desplegable["values"] = ('31.5', '63', '125', '250', '500', '1k', '2k', '4k', '8k', '16k')
        desplegable.grid(row=1,column=4,padx=2,pady=2)
    else :
        return

          
    factor = np.power(2,1.0/6.0)
    
    frecuenciaInferior = frecuenciaCentral/factor;
    frecuenciaSuperior = frecuenciaCentral*factor;
    audioFiltrado = []
    audioFiltradoL = []
    audioFiltradoR = []
    for inferior,superior in zip(frecuenciaInferior, frecuenciaSuperior):
        sos = signal.butter(N=3, Wn=np.array([ inferior, 
        superior])/fs, btype='bandpass', analog=False, 
        output='sos');
        if varBoton_Channels.get()==3:
            audioFiltrado.append(signal.sosfilt(sos, RIR_final))
        elif varBoton_Channels.get()==1:
            audioFiltradoL.append(signal.sosfilt(sos, RIR_finalL))
            audioFiltradoR.append(signal.sosfilt(sos, RIR_finalR))
            audioFiltrado.append(signal.sosfilt(sos, RIR_finalL))
        else:
            audioFiltradoL.append(signal.sosfilt(sos, RIR_finalL))
            audioFiltradoR.append(signal.sosfilt(sos, RIR_finalR))
            audioFiltrado.append(signal.sosfilt(sos, RIR_finalR))
    
    if varBoton_Channels.get()==3:
        audioFiltradoL=audioFiltrado
        audioFiltradoR=audioFiltrado
    
filtrado()    

#-----  Moving Median


def medianaMovil():
    # This function smooths the Impulse Response by the Moving Median Filter
    
    global medianaMovil_ET, medianaMovil_AT, energiaMediana, t_MM
    if varBotonSuavizado_MM.get() == 1 and cuadro_ventanaMM.get()== "":
        messagebox.showinfo("RIRs Prossesing Software", "You have to choose a window size")
        varBotonSuavizado_MM.set(0)
    if varBotonSuavizado_MM.get() == 1 and varBotonSuavizado_Sch.get() == 0:
        ventana=int(cuadro_ventanaMM.get())
        medianaMovil_ET=[]
        medianaMovil_AT = []
        energiaMediana=[]
        t_MM = []
        for i in range(len(audioFiltrado)):
            energiaMediana.append(10*np.log10((audioFiltrado[i]**2)/max(audioFiltrado[i]**2)))
            medianaMovil_ET.append(ndimage.median_filter(energiaMediana[i], size=ventana))
            medianaMovil_AT.append(np.sqrt((10**(medianaMovil_ET[i]/10))*max(medianaMovil_ET[i]**2)))
            t_MM.append(np.arange(len(medianaMovil_ET[i]))/fs)
    elif varBotonSuavizado_MM.get() == 1 and varBotonSuavizado_Sch.get() == 1:
        ventana=int(cuadro_ventanaMM.get())
        medianaMovil_ET=[]
        medianaMovil_AT = []
        energiaMediana=[]
        t_MM = []
        for i in range(len(audioFiltrado)):
            energiaMediana.append(10*np.log10((audioFiltrado[i]**2)/max(audioFiltrado[i]**2)))
            medianaMovil_ET.append(ndimage.median_filter(energiaMediana[i], size=ventana))
            medianaMovil_AT.append(np.sqrt((10**(medianaMovil_ET[i]/10))*max(medianaMovil_ET[i]**2)))
            t_MM.append(np.arange(len(medianaMovil_ET[i]))/fs)
        Schroeder()
    else:
        return

medianaMovil()


#---------- Schroeder by Chu and Lundeby  

#------Lundeby 

def lundeby(y,Fs):
    
    global punto,C
    punto = []
    C= []
    for i in range (len(y)):
        energia_impulso = y[i]**2
        
        # Calculates the noise level of the last 10% of the signal
        
        rms_dB = 10*np.log10(np.mean(energia_impulso[round(.9*len(energia_impulso)):])/max(energia_impulso))
        
        # Divides the intervals and calulates the mean
        
        t = math.floor(len(energia_impulso)/Fs/0.01)
        v = math.floor(len(energia_impulso)/t)
        
        media = []
        eje_tiempo = []
        for n in range (1,t):
            media.append(np.mean(energia_impulso[(((n-1)*v)+1):(n*v)]))
            eje_tiempo.append(math.ceil(v/2)+((n-1)*v))
        
        mediadB = 10*np.log10(media/max(energia_impulso))
        
        # It obtains the lineal regression between 0db and the mean more close to the mean+10 dB
        r = np.where(mediadB > rms_dB+10)
        r = max(r[0]) 
        A,B = stats.linregress(eje_tiempo[1:r], mediadB[1:r])[0:2]
        cruce = (rms_dB-B)/A
        
        if rms_dB > -20 :   # Insufficient SNR
            punto.append(len(energia_impulso))
            C.append(0)
       
        else:
            
        #-----IT BEGINS THE ITERATIVE PROCESS
        
            error=1
            INTMAX=50
            veces=1
            
            while error > 0.0001 and veces <= INTMAX:
            
               
                # Calculates the new vector of time for the mean  
                del t, v, n, media, eje_tiempo
                t = {}
                noise = np.array([])
                p = 5                           # Number of steps every 10 dB
                delta = abs(10/A)               # Number of samples for the 10 dB decay line
                v = math.floor(delta/p)         # Interval for obtaining the mean
                if (cruce-delta)>len(energia_impulso):
                    t=math.floor(len(energia_impulso)/v)
                else:
                    t = math.floor(len(energia_impulso[1:round(cruce-delta)])/v)
                    
                if t < 2:                        # Number of intervals to obtain the new mean in the interval 
                    t=2                          # ranging from the start to 10 dB before the crossover point
                elif bool(t) == False: 
                    t=2
                media = []
                eje_tiempo = []
                for n in range(1,t):
                    media.append(np.mean(energia_impulso[(((n-1)*v)+1):(n*v)]))
                    eje_tiempo.append(math.ceil(v/2)+((n-1)*v))
                
                mediadB = 10*np.log10(media/max(energia_impulso))
            
                del A, B, noise, rms_dB
                A,B = stats.linregress(eje_tiempo,mediadB)[0:2]
            
                # New mean noise energy, starting at the point of the decay line 10 dB below the crossover
                noise = energia_impulso[round(abs(cruce+delta)):]
                if (len(noise) < round(.1*len(energia_impulso))):
                    noise = energia_impulso[round(.9*len(energia_impulso)):] 
                       
                rms_dB = 10*np.log10(np.mean(noise)/max(energia_impulso))
        
                # New crossing point
                error = abs(cruce - (rms_dB-B)/A)/cruce
                cruce = round((rms_dB-B)/A)
                veces = veces + 1
            
        
        
    
        if cruce > len(energia_impulso):       # If the signal does not reach the noise level
            punto.append(len(energia_impulso))   # in the samples supplied, it is considered
        else :                                   # the crossing point the last sample,
            punto.append(cruce)                     # equivalent to not truncating the signal
            
        C.append(max(energia_impulso)*10**(B/10)*np.exp(A/10/np.log10(np.exp(1))*cruce)/(-A/10/np.log10(np.exp(1))))



#----- Shroeder

def Schroeder():
    global db, t_db, energiaSchroeder, t_es
    
    if varBotonSuavizado_Sch.get()==1 and varBotonSuavizado_MM.get() == 0:
        lundeby(audioFiltrado,fs)
        schrr = []
        db = []
        t_db = []
        energiaSchroeder=[]
        t_es=[]
        for i in range(len(audioFiltrado)):
            y = audioFiltrado[i]
            y=y[0:punto[i]]**2
            sch=audioFiltrado[i]**2 
            rms=np.mean(np.round(sch[int(.9*len(sch)):])) 
            schrr.append(np.flip((np.cumsum(np.flip(y-rms))+C[i])/(np.sum(y-rms)+C[i]))) 
            db.append(10*np.log10(np.abs(schrr[i]))) 
            t_db.append(np.arange(len(db[i]))/fs)
            energiaSchroeder.append(10*np.log10((audioFiltrado[i]**2)/max(audioFiltrado[i]**2)))
            t_es.append(np.arange(len(audioFiltrado[i]))/fs)
        
    elif varBotonSuavizado_Sch.get()==1 and varBotonSuavizado_MM.get() == 1:
        lundeby(medianaMovil_AT,fs)
        schrr = []
        db = []
        t_db = []
        energiaSchroeder=[]
        t_es=[]
        for i in range(len(medianaMovil_AT)):
            y = medianaMovil_AT[i]
            y=y[0:punto[i]]**2
            sch=medianaMovil_AT[i]**2 
            rms=np.mean(np.round(sch[int(.9*len(sch)):])) 
            schrr.append(np.flip((np.cumsum(np.flip(y-rms))+C[i])/(np.sum(y-rms)+C[i]))) 
            db.append(10*np.log10(np.abs(schrr[i]))) 
            t_db.append(np.arange(len(db[i]))/fs)
            energiaSchroeder.append(10*np.log10((audioFiltrado[i]**2)/max(audioFiltrado[i]**2)))
            t_es.append(np.arange(len(audioFiltrado[i]))/fs)
    else:
        return

    
Schroeder()

#----- Parameters

def parametros():
    global EDT, T10, T20, T30, c50, c80
    global Tt, EDTt, IACCearly, db
    
    # Variables for EDT, T10, T20, T30 
    
    initT = -5.0   
    
    # Variables T30
    endT30 = -35.0
    factorT30 = 2.0
    
    # Variables T20
    endT20 = -25.0
    factorT20 = 3.0
    
    # Variables T10
    endT10 = -15.0
    factorT10 = 6.0
    
    # Variables EDT
    initEDT = 0.0
    endEDT = -10.0
    factorEDT = 6.0
    
    # Clarity 
    c50=[]
    c80=[]
    time50=50 
    time80=80
    
    # Tt
    Tt = []
    Tt_indice = []
    
    # Linear regresion 
    LR_EDT = []
    LR_T10 = []
    LR_T20 = []
    LR_T30 = []
    LR_EDTt = []
    slopeListEDT = []
    slopeListT10 = []
    slopeListT20 = []
    slopeListT30 = []
    slopeListEDTt = []
    EDT = []
    T10 = []
    T20 = []
    T30 = []
    EDTt = []
    
    # IACCearly
    IACCearly=[]
    
    db_MM = []
    t_dbMM = []
    
    # It's used when it aplies Moving Median only 
    
    if varBotonSuavizado_MM.get()==1 and varBotonSuavizado_Sch.get()==0:
        for i in range(len(audioFiltrado)):
            cortamos = medianaMovil_ET[i]
            cortamos = cortamos[cortamos.argmax():]
            dif = 0-max(cortamos)
            cortamos = cortamos + dif
            t_cortamos = np.arange(len(cortamos))/fs
            db_MM.append(cortamos)
            t_dbMM.append(t_cortamos)

    # Here is when all parameters are calculated 
    
    for i in range(len(audioFiltrado)):
        
        # IACCearly
        IACF=[]
        PL=audioFiltradoL[i]
        PR=audioFiltradoR[i]
        PLsqr=PL**2
        PRsqr=PR**2
        Tau = np.arange(0,44,1) # This vector corresponds to the vector that goes between 0, 1 ms
        Tearly=int(((80*fs)/1000)+1) # Final time in [ms] 
        for q in range(len(Tau)):
            IACF.append(np.sum(PL[:Tearly]*PR[Tau[q]:Tearly+Tau[q]])/np.sqrt(np.sum(PLsqr[:Tearly])*np.sum(PRsqr[:Tearly])))
        IACCearly.append(round(max(IACF),2))
        
        # Clarity
        energiaBanda = audioFiltrado[i]**2
        t50 = int((time50 / 1000) * fs + 1)
        t80 = int((time80 / 1000) * fs + 1) 
        c50.append(round(10* np.log10((np.sum(energiaBanda[:t50]) / np.sum(energiaBanda[t50:]))),3))
        c80.append(round(10* np.log10((np.sum(energiaBanda[:t80]) / np.sum(energiaBanda[t80:]))),3))
        
        # Transition Time
        energiaBanda=energiaBanda[np.argmax(energiaBanda):]
        u=0
        while energiaBanda[u]>energiaBanda[u+1]:
            u=u+1    
        sinSonidoDirecto=energiaBanda[u:]
        total=sum(sinSonidoDirecto)
        totalParcial=np.cumsum(sinSonidoDirecto)
        indice=(np.abs(totalParcial - 0.99*total).argmin()) 
        Tt_indice.append(indice+u+10)
        Tt.append(round(Tt_indice[i]/fs,3))
        
        # RT's
        
        if varBotonSuavizado_MM.get()==1 and varBotonSuavizado_Sch.get()==0:
            db = db_MM
            
        energiaTt = db[i]
        sch_initEDT = np.abs(db[i] - initEDT).argmin()
        sch_initT = np.abs(db[i] - initT).argmin()
        sch_endEDT = np.abs(db[i] - endEDT).argmin()
        sch_endT10 = np.abs(db[i] - endT10).argmin()
        sch_endT20 = np.abs(db[i] - endT20).argmin()
        sch_endT30 = np.abs(db[i] - endT30).argmin()
        sch_endEDTt = np.abs(db[i] - energiaTt[Tt_indice[i]]).argmin()

        xEDT = np.arange(sch_initEDT, sch_endEDT + 1) / fs
        xT10 = np.arange(sch_initT, sch_endT10 + 1) / fs
        xT20 = np.arange(sch_initT, sch_endT20 + 1) / fs
        xT30 = np.arange(sch_initT, sch_endT30 + 1) / fs
        xEDT = np.arange(sch_initEDT, sch_endEDT + 1) / fs
        xEDTt = np.arange(sch_initEDT, sch_endEDTt + 1) / fs        
        y = db[i]
        yEDT = y[sch_initEDT:sch_endEDT + 1]
        yT10 = y[sch_initT:sch_endT10 + 1]
        yT20 = y[sch_initT:sch_endT20 + 1]
        yT30 = y[sch_initT:sch_endT30 + 1]
        yEDTt = y[sch_initEDT:sch_endEDTt + 1]
        
        slopeEDT,interceptEDT = stats.linregress(xEDT, yEDT)[0:2]
        slopeT10,interceptT10 = stats.linregress(xT10, yT10)[0:2]
        slopeT20,interceptT20 = stats.linregress(xT20, yT20)[0:2]
        slopeT30,interceptT30 = stats.linregress(xT30, yT30)[0:2]
        slopeEDTt,interceptEDTt = stats.linregress(xEDTt, yEDTt)[0:2]
        
        slopeListEDT.append(slopeEDT)
        slopeListT10.append(slopeT10)
        slopeListT20.append(slopeT20)
        slopeListT30.append(slopeT30)
        slopeListEDTt.append(slopeEDTt)
        
        EDT.append(round(-60/slopeListEDT[i],3))
        T10.append(round(-60/slopeListT10[i],3))
        T20.append(round(-60/slopeListT20[i],3))
        T30.append(round(-60/slopeListT30[i],3))
        EDTt.append(round(-60/slopeListEDTt[i],3))
        
        LR_EDT.append(slopeEDT*t[0:44100] + interceptEDT)
        LR_T10.append(slopeT10*t[0:44100] + interceptT10)
        LR_T20.append(slopeT20*t[0:44100] + interceptT20)
        LR_T30.append(slopeT30*t[0:44100] + interceptT30)
        LR_EDTt.append(slopeEDTt*t[0:44100] + interceptEDTt)


#----- Table parameters

area=('#', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-')
ac=('all','n','e','s','ne','nw','sw','c','a','b','h')
ad=('all','n','e','s','ne','nw','sw','c')
sales_data=[('EDT','','','','', '','',''),
            ('EDTt','','','','', '','',''),
            ('T10','','','','', '','',''),
            ('T20','','','','', '','',''),
            ('T30','','','','', '','',''),
            ('C50','','','','', '','',''),
            ('C80','','','','', '','',''),
            ('Tt','','','','', '','',''),
            ('IACC','','','','', '','','')
            ]

tv=ttk.Treeview(miFrame,columns=ac,show='headings',height=7)
tv.grid(row=8,column=3,padx=10,pady=10,rowspan=4)



for i in range(len(area)):
    tv.column(ac[i],width=70,anchor='e')
    tv.heading(ac[i],text=area[i])

for i in range(len(sales_data)):
    tv.insert('','end',values=sales_data[i])

# ----- Interactive Table
 
def tabla():
    
    # Default Table and Graph
    
    parametros()
    
    if varBotonFiltro.get()==1:
        area=('#','250', '315','400', '500', '630', '800', '1000', '1250', '1600', '2000')
        ac=('all','n','e','s','ne','nw','sw','c','a','b','sr')
        sales_data=[('EDT',EDT[10],EDT[11],EDT[12],EDT[13],EDT[14],EDT[15],EDT[16],EDT[17],EDT[18],EDT[19]),
                    ('EDTt',EDTt[10],EDTt[11],EDTt[12],EDTt[13],EDTt[14],EDTt[15],EDTt[16],EDTt[17],EDTt[18],EDTt[19]),
                    ('T10',T10[10],T10[11],T10[12],T10[13],T10[14],T10[15],T10[16],T10[17],T10[18],T10[19]),
                    ('T20',T20[10],T20[11],T20[12],T20[13],T20[14],T20[15],T20[16],T20[17],T20[18],T20[19]),
                    ('T30',T30[10],T30[11],T30[12],T30[13],T30[14],T30[15],T30[16],T30[17],T30[18],T30[19]),
                    ('C50',c50[10],c50[11],c50[12],c50[13],c50[14],c50[15],c50[16],c50[17],c50[18],c50[19]),
                    ('c80',c80[10],c80[11],c80[12],c80[13],c80[14],c80[15],c80[16],c80[17],c80[18],c80[19]),
                    ('Tt[Seg]',Tt[10],Tt[11],Tt[12],Tt[13],Tt[14],Tt[15],Tt[16],Tt[17],Tt[18],Tt[19]),
                    ('IACCearly',IACCearly[10],IACCearly[11],IACCearly[12],IACCearly[13],IACCearly[14],IACCearly[15],
                     IACCearly[16],IACCearly[17],IACCearly[18],IACCearly[19])
                    ]
        tv=ttk.Treeview(miFrame,columns=ac, show='headings',height=7)
        tv.grid(row=8,column=3,padx=10,pady=10,rowspan=4)
        
    
        for i in range(len(area)):
            tv.column(ac[i],width=70,anchor='e')
            tv.heading(ac[i],text=area[i])

        for i in range(len(sales_data)):
            tv.insert('','end',values=sales_data[i])
    
    elif varBotonFiltro.get()==2:
        area=('#','31.5', '63', '125', '250', '500', '1000', '2000', '4000', '8000', '16000')
        ac=('all','n','e','s','ne','nw','sw','c','a','b','d')
        sales_data=[('EDT',EDT[0],EDT[1],EDT[2],EDT[3],EDT[4],EDT[5],EDT[6],EDT[7],EDT[8],EDT[9]),
                    ('EDTt',EDTt[0],EDTt[1],EDTt[2],EDTt[3],EDTt[4],EDTt[5],EDTt[6],EDTt[7],EDTt[8],EDTt[9]),
                    ('T10',T10[0],T10[1],T10[2],T10[3],T10[4],T10[5],T10[6],T10[7],T10[8],T10[9]),
                    ('T20',T20[0],T20[1],T20[2],T20[3],T20[4],T20[5],T20[6],T20[7],T20[8],T20[9]),
                    ('T30',T30[0],T30[1],T30[2],T30[3],T30[4],T30[5],T30[6],T30[7],T30[8],T30[9]),
                    ('C50',c50[0],c50[1],c50[2],c50[3],c50[4],c50[5],c50[6],c50[7],c50[8],c50[9]),
                    ('C80',c80[0],c80[1],c80[2],c80[3],c80[4],c80[5],c80[6],c80[7],c80[8],c80[9]),
                    ('Tt[Seg]',Tt[0],Tt[1],Tt[2],Tt[3],Tt[4],Tt[5],Tt[6],Tt[7],Tt[8],Tt[9]),
                    ('IACCearly',IACCearly[0],IACCearly[1],IACCearly[2],IACCearly[3],IACCearly[4],IACCearly[5],
                     IACCearly[6],IACCearly[7],IACCearly[8],IACCearly[9])
                    ]
        tv=ttk.Treeview(miFrame,columns=ac, show='headings',height=7)
        tv.grid(row=8,column=3,padx=10,pady=10,rowspan=4)
    
        for i in range(len(area)):
            tv.column(ac[i],width=70,anchor='e')
            tv.heading(ac[i],text=area[i])

        for i in range(len(sales_data)):
            tv.insert('','end',values=sales_data[i])
    else:
        return
    
    fig = plt.Figure(figsize=(7.7,4), dpi=100)
    fig.add_subplot(111).set_title("Impulse Response for 1k Hz")
    fig.add_subplot(111).set_xlabel("Time [s]")
    fig.add_subplot(111).set_ylabel("Level [dB]")
    if varBotonFiltro.get()==1:
        indiceDefault=16
    else:
        indiceDefault=5
    if varBotonSuavizado_Sch.get()==1 and varBotonSuavizado_MM.get() == 1:
        fig.add_subplot(111).plot(t_es[indiceDefault],energiaSchroeder[indiceDefault],label='Energy Signal')
        fig.add_subplot(111).plot(t_es[indiceDefault],medianaMovil_ET[indiceDefault],label='Moving Median')
        fig.add_subplot(111).plot(t_db[indiceDefault],db[indiceDefault],label='Schroeder')
    elif varBotonSuavizado_Sch.get()==1 and varBotonSuavizado_MM.get() == 0:
        fig.add_subplot(111).plot(t_es[indiceDefault],energiaSchroeder[indiceDefault],label='Energy Signal')
        fig.add_subplot(111).plot(t_db[indiceDefault],db[indiceDefault],label='Schroeder')
    elif varBotonSuavizado_Sch.get()==0 and varBotonSuavizado_MM.get() == 1:
        fig.add_subplot(111).plot(t_MM[indiceDefault],medianaMovil_ET[indiceDefault],label='Moving Median')
    else:
        return
    fig.add_subplot(111).legend()
    grafico = FigureCanvasTkAgg(fig,miFrame) 
    grafico.get_tk_widget().grid(row=1,column=3,padx=10,pady=10,rowspan=7)
    
    
def refreshTabla():

# When we want to refresh the table
    
    global idx
    if varBotonFiltro.get()==1:
        for i in range(len(desplegable2["values"])):
            
            if desplegable2["values"][i]==seleccion2.get():
                if i==0:
                    idx=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                    area=('#','25', '31.5', '40', '50', '63', '80', '100', '125', '160', '200')
                elif i==1:
                    idx=[10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
                    area=('#','250', '315','400', '500', '630', '800', '1000', '1250', '1600', '2000')
                else:
                    idx=[19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
                    area=('#','2000', '2500', '3150', '4000','5000', '6300', '8000', '10000', '12500', '16000')

        ac=('all','n','e','s','ne','nw','sw','c','a','b','d')
        sales_data=[('EDT',EDT[idx[0]],EDT[idx[1]],EDT[idx[2]],EDT[idx[3]],EDT[idx[4]],EDT[idx[5]],EDT[idx[6]],EDT[idx[7]],EDT[idx[8]],EDT[idx[9]]),
                    ('EDTt',EDTt[idx[0]],EDTt[idx[1]],EDTt[idx[2]],EDTt[idx[3]],EDTt[idx[4]],EDTt[idx[5]],EDTt[idx[6]],EDTt[idx[7]],EDTt[idx[8]],EDTt[idx[9]]),
                    ('T10',T10[idx[0]],T10[idx[1]],T10[idx[2]],T10[idx[3]],T10[idx[4]],T10[idx[5]],T10[idx[6]],T10[idx[7]],T10[idx[8]],T10[idx[9]]),
                    ('T20',T20[idx[0]],T20[idx[1]],T20[idx[2]],T20[idx[3]],T20[idx[4]],T20[idx[5]],T20[idx[6]],T20[idx[7]],T20[idx[8]],T20[idx[9]]),
                    ('T30',T30[idx[0]],T30[idx[1]],T30[idx[2]],T30[idx[3]],T30[idx[4]],T30[idx[5]],T30[idx[6]],T30[idx[7]],T30[idx[8]],T30[idx[9]]),
                    ('C50',c50[idx[0]],c50[idx[1]],c50[idx[2]],c50[idx[3]],c50[idx[4]],c50[idx[5]],c50[idx[6]],c50[idx[7]],c50[idx[8]],c50[idx[9]]),
                    ('C80',c80[idx[0]],c80[idx[1]],c80[idx[2]],c80[idx[3]],c80[idx[4]],c80[idx[5]],c80[idx[6]],c80[idx[7]],c80[idx[8]],c80[idx[9]]),
                    ('Tt[Seg]',Tt[idx[0]],Tt[idx[1]],Tt[idx[2]],Tt[idx[3]],Tt[idx[4]],Tt[idx[5]],Tt[idx[6]],Tt[idx[7]],Tt[idx[8]],Tt[idx[9]]),
                    ('IACCearly',IACCearly[idx[0]],IACCearly[idx[1]],IACCearly[idx[2]],IACCearly[idx[3]],IACCearly[idx[4]],IACCearly[idx[5]],
                     IACCearly[idx[6]],IACCearly[idx[7]],IACCearly[idx[8]],IACCearly[idx[9]])
                    ]
        tv=ttk.Treeview(miFrame,columns=ac, show='headings',height=7)
        tv.grid(row=8,column=3,padx=10,pady=10,rowspan=4)
    
        for i in range(len(area)):
            tv.column(ac[i],width=70,anchor='e')
            tv.heading(ac[i],text=area[i])
    
        for i in range(len(sales_data)):
            tv.insert('','end',values=sales_data[i])
        
    else: 
        return

refreshTabla()

#----- Graphs 

figInicial = plt.Figure(figsize=(7.7,4), dpi=100)
figInicial.add_subplot(111).set_title("Impulse Response")
figInicial.add_subplot(111).set_xlabel("Time [s]")
figInicial.add_subplot(111).set_ylabel("Level [dB]")
figInicial.add_subplot(111).plot()

graficoInicial = FigureCanvasTkAgg(figInicial,miFrame) 
graficoInicial.get_tk_widget().grid(row=1,column=3,padx=10,pady=10,rowspan=7)



def graficar():
    
    # This functions it's done so the user can interact with diferent band graphis
    
    if varBotonFiltro.get()==1 or varBotonFiltro.get()==2:
        for i in range(len(desplegable["values"])):
            if desplegable["values"][i]==seleccion.get():
                indiceGrafico = i
    else:
        return
    
    fig = plt.Figure(figsize=(7.7,4), dpi=100)
    fig.add_subplot(111).set_title("Impulse Response")
    fig.add_subplot(111).set_xlabel("Time [s]")
    fig.add_subplot(111).set_ylabel("Level [dB]")
    if varBotonSuavizado_Sch.get()==1 and varBotonSuavizado_MM.get() == 1:
        fig.add_subplot(111).plot(t_es[indiceGrafico],energiaSchroeder[indiceGrafico],label='Energy Signal')
        fig.add_subplot(111).plot(t_es[indiceGrafico],medianaMovil_ET[indiceGrafico],label='Moving Median')
        fig.add_subplot(111).plot(t_db[indiceGrafico],db[indiceGrafico],label='Schroeder')
    elif varBotonSuavizado_Sch.get()==1 and varBotonSuavizado_MM.get() == 0:
        fig.add_subplot(111).plot(t_es[indiceGrafico],energiaSchroeder[indiceGrafico],label='Energy Signal')
        fig.add_subplot(111).plot(t_db[indiceGrafico],db[indiceGrafico],label='Schroeder')
    elif varBotonSuavizado_Sch.get()==0 and varBotonSuavizado_MM.get() == 1:
        fig.add_subplot(111).plot(t_MM[indiceGrafico],medianaMovil_ET[indiceGrafico],label='Moving Median')
    else:
        return
    fig.add_subplot(111).legend()
    grafico = FigureCanvasTkAgg(fig,miFrame) 
    grafico.get_tk_widget().grid(row=1,column=3,padx=10,pady=10,rowspan=7)

#----- Export table 
    
def exportar():
    #This function is done so the user can export the parameters table
    
    global parametrosExcel
    if varBotonFiltro.get()==2:
        filasXlsx=[EDT,EDTt,T10,T20,T30,c50,c80,Tt,IACCearly]
        DATA=pd.DataFrame(filasXlsx,columns=['31.5', '63', '125', '250', '500', '1000', '2000', '4000', '8000', '16000'])
        parametrosExcel = DATA.to_excel('Parametros.xlsx')
    elif varBotonFiltro.get()==1:
        filasXlsx=[EDT,EDTt,T10,T20,T30,c50,c80,Tt,IACCearly]
        DATA=pd.DataFrame(filasXlsx,columns=['25', '31.5', '40', '50', '63', '80', '100', '125', '160', '200', '250', '315',
              '400', '500', '630', '800', '1000', '1250', '1600', '2000', '2500', '3150', '4000',
              '5000', '6300', '8000', '10000', '12500', '16000'])
        parametrosExcel = DATA.to_excel('Parametros.xlsx')
    else:
        return

exportar()

#---- Design callback

def elBotonSine():
    # This function makes appear the fill boxes so the user can income Sine Sweep parameters
    
    global cuadroDuracion, cuadroFrecuencia_min, cuadroFrecuencia_max
    if varBotonArchivo.get()==1:
        duracion.config(bg="#18232d",fg="white",justify="right")
        frecuenciaMin.config(bg="#18232d",fg="white",justify="center")
        frecuenciaMax.config(bg="#18232d",fg="white",justify="center")
        cuadroDuracion = Entry(miFrame)
        cuadroDuracion.grid(row=4,column=1,padx=2,pady=2,columnspan=2) 
        cuadroDuracion.config(width=8,bg="#18232d",fg="white",justify="center")
        cuadroFrecuencia_min = Entry(miFrame)
        cuadroFrecuencia_min.grid(row=5,column=1,padx=2,pady=2,columnspan=2)
        cuadroFrecuencia_min.config(width=8,bg="#18232d",fg="white",justify="center")
        cuadroFrecuencia_max = Entry(miFrame)
        cuadroFrecuencia_max.grid(row=6,column=1,padx=2,pady=2,columnspan=2)
        cuadroFrecuencia_max.config(width=8,bg="#18232d",fg="white",justify="center")
    elif varBotonArchivo.get()==2:
        duracion.config(bg="#18232d",fg="#18232d",justify="right")
        frecuenciaMin.config(bg="#18232d",fg="#18232d",justify="center")
        frecuenciaMax.config(bg="#18232d",fg="#18232d",justify="center")
        cuadroDuracion.destroy()
        cuadroFrecuencia_min.destroy()
        cuadroFrecuencia_max.destroy()

    else:
        return

elBotonSine()

#%% Blocks and Texts
 
titulo = Label(miFrame,text="RIRs Processing Software",font=("Times New Roman",24))
titulo.grid(row=0,column=0,columnspan=5)
titulo.config(bg="#18232d",fg="white",justify="center")

duracion = Label(miFrame,text="Duration:")
duracion.config(bg="#18232d",fg="#18232d",justify="right")
duracion.grid(row=4,column=0,padx=104,pady=2,columnspan=2) 

frecuenciaMin = Label(miFrame,text="Minimum Frequency:")
frecuenciaMin.config(bg="#18232d",fg="#18232d",justify="center")
frecuenciaMin.grid(row=5,column=0,padx=2,pady=2,columnspan=2) 


frecuenciaMax = Label(miFrame,text="Maximum Frequency:")
frecuenciaMax.config(bg="#18232d",fg="#18232d",justify="center")
frecuenciaMax.grid(row=6,column=0,padx=2,pady=2,columnspan=2) 


ventanaMM = Label(miFrame,text="Moving Median Window")
ventanaMM.config(bg="#18232d",fg="white",justify="center",font=("Arial",8))
ventanaMM.grid(row=9,column=0,sticky="w",padx=2,pady=2,columnspan=2)
muestrasMM = Label(miFrame,text="# Samples:")
muestrasMM.config(bg="#18232d",fg="white",justify="center")
muestrasMM.grid(row=10,column=0,padx=2,pady=2) 
cuadro_ventanaMM = Entry(miFrame)
cuadro_ventanaMM.grid(row=10,column=1,padx=2,pady=2)
cuadro_ventanaMM.config(bg="#18232d",fg="white",justify="center")


#%% Buttons

botonLoad = Button(miFrame, text="Load File",command=abrirFichero)
botonLoad.grid(row=1,column=1,padx=2,pady=2)
botonLoad.config(justify="center")

botonCalcular = Button(miFrame, text="CALCULATE", command=tabla)
botonCalcular.grid(row=11,column=2,sticky="e",padx=2,pady=2)
botonCalcular.config(justify="center")

botonGrafico = Button(miFrame, text="Refresh Graph", command=graficar)
botonGrafico.grid(row=7,column=4,sticky="w",padx=2,pady=2)
botonGrafico.config(justify="center")

botonTabla = Button(miFrame, text="Refresh Table", command=refreshTabla)
botonTabla.grid(row=10,column=4,sticky="w",padx=2,pady=2)
botonTabla.config(justify="center")

botonExportar = Button(miFrame, text="Export Table", command=exportar)
botonExportar.grid(row=11,column=4,sticky="w",padx=2,pady=2)
botonExportar.config(justify="center")

Radiobutton(miFrame, text="Third Band Filter", variable=varBotonFiltro,value=1,
            indicator=0,command=filtrado).grid(row=7, column=0,padx=2,pady=2,columnspan=2)
Radiobutton(miFrame, text="Band Filter", variable=varBotonFiltro,value=2,
            indicator=0,command=filtrado).grid(row=7, column=1,padx=2,pady=2,columnspan=2)

Radiobutton(miFrame, text="Sine Sweep",variable=varBotonArchivo,value=1,
            indicator=0,command=elBotonSine).grid(row=3,column=0,padx=2,pady=2,columnspan=2)
Radiobutton(miFrame,text="Impulse Response",variable=varBotonArchivo,
            value=2,indicator=0,command=elBotonSine).grid(row=3,column=1,padx=2,pady=2,columnspan=2)

Radiobutton(miFrame,width=4, text="Left", variable=varBoton_Channels,value=1,indicator=0,
            command=lecturaArchivos).grid(row=2, column=0,sticky="e",padx=2,pady=2)
Radiobutton(miFrame, text="Mono", variable=varBoton_Channels,value=3,indicator=0,
            command=lecturaArchivos).grid(row=2, column=1,padx=2,pady=2)
Radiobutton(miFrame, text="Right", variable=varBoton_Channels,value=2,indicator=0,
            command=lecturaArchivos).grid(row=2, column=2,sticky="w",padx=2,pady=2)

Checkbutton(miFrame,text="Moving Median",variable=varBotonSuavizado_MM,onvalue=1,
            offvalue=0,command=medianaMovil,bg="#18232d",fg="white",selectcolor="#18232d").grid(row=8, column=0,padx=2,pady=2,columnspan=2)
Checkbutton(miFrame,text="Schroeder",variable=varBotonSuavizado_Sch,onvalue=1,
            offvalue=0,command=Schroeder,bg="#18232d",fg="white",selectcolor="#18232d").grid(row=8,column=1,padx=2,pady=2,columnspan=2)

desplegable = ttk.Combobox(miFrame)
desplegable.grid(row=1,column=4,padx=2,pady=2)

desplegable2 = ttk.Combobox(miFrame)
desplegable2.grid(row=8,column=4,padx=2,pady=2)

  
raiz.mainloop()