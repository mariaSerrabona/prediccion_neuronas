import numpy as np
import ficheros_poo.funciones_utiles as funciones_utiles

class programacion_percepcion_clase ():

    def __init__(self, observaciones_entradas, predicciones, epochs, sesgo, txAprendizaje, peso):
        self.observaciones_entradas=observaciones_entradas
        self.predicciones=predicciones
        self.epochs=epochs
        self.sesgo=sesgo
        self.txAprendizaje=txAprendizaje
        self.peso=peso

    #--------------------------------------
    #    APRENDIZAJE
    #--------------------------------------
    def aprendizaje (self):
        #--------------------------------------
        #       GRÁFICA
        #--------------------------------------

        Grafica_MSE=[]

        for epoch in range(0,self.epochs):
            print("EPOCH ("+str(epoch)+"/"+str(self.epochs)+")")
            predicciones_realizadas_durante_epoch = []
            predicciones_esperadas = []
            numObservacion = 0
            for observacion in self.observaciones_entradas:

                #Carga de la capa de entrada
                x1 = observacion[0]
                x2 = observacion[1]

                #Valor de predicción esperado
                valor_esperado = self.predicciones[numObservacion][0]
                wb=0
                #Etapa 1: Cálculo de la suma ponderada
                valor_suma_ponderada = funciones_utiles.funciones_utiles.suma_ponderada(x1,self.peso [1],x2,self.peso [2],self.sesgo,wb)


                #Etapa 2: Aplicación de la función de activación
                valor_predicho = funciones_utiles.funciones_utiles.funcion_activacion_sigmoide(valor_suma_ponderada)


                #Etapa 3: Cálculo del error
                valor_error =  funciones_utiles.funciones_utiles.error_lineal(valor_esperado,valor_predicho)


                #Actualización del peso 1
                #Cálculo ddel gradiente del valor de ajuste y del peso nuevo
                gradiente_W11 =  funciones_utiles.funciones_utiles.calculo_gradiente(x1,valor_predicho,valor_error)
                valor_ajuste_W11 =  funciones_utiles.funciones_utiles.calculo_valor_ajuste(gradiente_W11,self.txAprendizaje)
                w11 =  funciones_utiles.funciones_utiles.calculo_nuevo_peso(self.peso[1],valor_ajuste_W11)

                # Actualización del peso 2
                gradiente_W21 =  funciones_utiles.funciones_utiles.calculo_gradiente(x2, valor_predicho, valor_error)
                valor_ajuste_W21 =  funciones_utiles.funciones_utiles.calculo_valor_ajuste(gradiente_W21, self.txAprendizaje)
                w21 =  funciones_utiles.funciones_utiles.calculo_nuevo_peso(self.peso[2], valor_ajuste_W21)


                # Actualización del peso del sesgo
                gradiente_Wb =  funciones_utiles.funciones_utiles.calculo_gradiente(self.sesgo, valor_predicho, valor_error)
                valor_ajuste_Wb =  funciones_utiles.funciones_utiles.calculo_valor_ajuste(gradiente_Wb, self.txAprendizaje)
                wb =  funciones_utiles.funciones_utiles.calculo_nuevo_peso(wb, valor_ajuste_Wb)

                print("     EPOCH (" + str(self.epochs) + "/" + str(self.epochs) + ") -  Observación: " + str(numObservacion+1) + "/" + str(len( self.observaciones_entradas)))

                #Almacenamiento de la predicción realizada:
                predicciones_realizadas_durante_epoch.append(valor_predicho)
                predicciones_esperadas.append(self.predicciones[numObservacion][0])

                #Paso a la observación siguiente
                numObservacion = numObservacion+1

            MSE =  funciones_utiles.funciones_utiles.calculo_MSE(predicciones_realizadas_durante_epoch, self.predicciones, predicciones_esperadas)
            Grafica_MSE.append(MSE[0])
            print("MSE: "+str(MSE))



            import matplotlib.pyplot as plt
            plt.plot(Grafica_MSE)
            plt.ylabel('MSE')
            plt.show()


            print()
            print()
            print ("¡Aprendizaje terminado!")
            print ("Pesos iniciales: " )
            print ("W11 = "+str(self.peso[0]))
            print ("W21 = "+str(self.peso[1]))
            print ("Wb = "+str(self.peso[3]))

            print ("Pesos finales: " )
            print ("W11 = "+str(w11))
            print ("W21 = "+str(w21))
            print ("Wb = "+str(wb))

            print()
            print("--------------------------")
            print ("PREDICCIÓN ")
            print("--------------------------")
            x1 = 1
            x2 = 1

            #Etapa 1: Cálculo de la suma ponderada
            valor_suma_ponderada =  funciones_utiles.funciones_utiles.suma_ponderada(x1,w11,x2,w21,self.sesgo,wb)


            #Etapa 2: Aplicación de la función de activación
            valor_predicho =  funciones_utiles.funciones_utiles.funcion_activacion_sigmoide(valor_suma_ponderada)
            #valor_predicho = funcion_activacion_relu(valor_suma_ponderada)

            print("Predicción del [" + str(x1) + "," + str(x2)  + "]")
            print("Predicción = " + str(valor_predicho))


def main():

    observaciones_entradas = np.array([
                                [1, 0],
                                [1, 1],
                                [0, 1],
                                [0, 0]
                                ])


    predicciones = np.array([[0],[1], [0],[0]])


    #--------------------------------------
    #      PARAMETRIZACIÓN DEL PERCEPTRÓN
    #--------------------------------------

    #Generación de los pesos en el intervalo [-1;1]
    np.random.seed(1)
    limiteMin = -1
    limiteMax = 1

    w11 = (limiteMax-limiteMin) * np.random.random() + limiteMin
    w21 = (limiteMax-limiteMin) * np.random.random() + limiteMin
    w31 = (limiteMax-limiteMin) * np.random.random() + limiteMin

    #El sesgo
    sesgo = 1
    wb = 0

    #Almacenamiento de los pesos iniciales, solo para visualización al final del aprendizaje
    peso = [w11,w21,w31,wb]

    #Tasa de aprendizaje
    txAprendizaje = 0.1

    #Cantidad de épocas
    epochs = 3


    info_percepcion=programacion_percepcion_clase(observaciones_entradas, predicciones, epochs, sesgo, txAprendizaje, peso)
    return info_percepcion.aprendizaje()