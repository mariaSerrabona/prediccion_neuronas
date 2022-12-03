import tensorflow as tf



class prediccion_tensorflow():

    def __init__(self, valores_entradas_X, valores_entradas_Y, tf_valores_reales_Y, tf_neuronas_entradas_X, peso, sesgo):
        self.valores_entradas_X = valores_entradas_X
        self.valores_a_predecir_Y=valores_entradas_Y
        self.tf_valores_reales_Y=tf_valores_reales_Y
        self.tf_neuronas_entradas_X=tf_neuronas_entradas_X
        self.peso=peso
        self.sesgo=sesgo



    def aprendizaje(self):

        sumaponderada = tf.matmul(self.tf_neuronas_entradas_X,self.peso)

        #Adición del sesgo a la suma ponderada
        sumaponderada = tf.add(sumaponderada,self.sesgo)

        #Función de activación de tipo sigmoide que permite calcular la predicción
        prediccion = tf.sigmoid(sumaponderada)

        #Función de error de media cuadrática MSE
        funcion_error = tf.reduce_sum(tf.pow(self.tf_valores_reales_Y-prediccion,2))

        #Descenso de gradiente con una tasa de aprendizaje fijada a 0,1
        optimizador = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1).minimize(funcion_error)

            #-------------------------------------
            #    APRENDIZAJE
            #-------------------------------------

            #Cantidad de epochs
        epochs = 10000

            #Inicialización de la variable
        init = tf.compat.v1.global_variables_initializer()

            #Inicio de una sesión de aprendizaje
        sesion = tf.compat.v1.Session()
        sesion.run(init)

            #Para la realización de la gráfica para la MSE
        Grafica_MSE=[]


            #Para cada epoch
        for i in range(epochs):

                #Realización del aprendizaje con actualzación de los pesos
            sesion.run(optimizador, feed_dict = {self.tf_neuronas_entradas_X: self.valores_entradas_X, self.tf_valores_reales_Y:self.valores_a_predecir_Y})

                #Calcular el error
            MSE = sesion.run(funcion_error, feed_dict = {self.tf_neuronas_entradas_X: self.valores_entradas_X, self.tf_valores_reales_Y:self.valores_a_predecir_Y})

                #Visualización de la información
            Grafica_MSE.append(MSE)
            print("EPOCH (" + str(i) + "/" + str(epochs) + ") -  MSE: "+ str(MSE))


            #Visualización gráfica
        import matplotlib.pyplot as plt
        plt.plot(Grafica_MSE)
        plt.ylabel('MSE')
        plt.show()

        print("--- VERIFICACIONES ----")

        for i in range(0,4):
            print("Observación:"+str(self.valores_entradas_X[i])+ " - Esperado: "+str(self.valores_a_predecir_Y[i])+" - Predicción: "+str(sesion.run(prediccion, feed_dict={self.tf_neuronas_entradas_X: [self.valores_entradas_X[i]]})))



        sesion.close()



def main():

    valores_entradas_X = [[1., 0.], [1., 1.], [0., 1.], [0., 0.]]
    valores_a_predecir_Y = [[0.], [1.], [0.], [0.]]
    #Variable TensorFLow correspondiente a los valores de neuronas de entrada
    tf_neuronas_entradas_X = tf.compat.v1.placeholder(tf.float32, [None, 2])

    #Variable TensorFlow correspondiente a la neurona de salida (predicción real)
    tf_valores_reales_Y = tf.compat.v1.placeholder(tf.float32, [None, 1])

    #-- Peso --
    #Creación de una variable TensorFlow de tipo tabla
    #que contiene 2 entradas y cada una tiene un peso [2,1]
    #Estos valores se inicializan al azar
    peso = tf.Variable(tf.random.normal([2, 1]), tf.float32)

    #-- Sesgo inicializado a 0 --
    sesgo = tf.Variable(tf.zeros([1, 1]), tf.float32)

    #La suma ponderada es en la práctica una multiplicación de matrices
    #entre los valores en la entrada X y los distintos pesos
    #la función matmul se encarga de hacer esta multiplicación
    sumaponderada = tf.compat.v1.matmul(tf_neuronas_entradas_X,peso)

    #Adición del sesgo a la suma ponderada
    sumaponderada = tf.add(sumaponderada,sesgo)


    neuronas=prediccion_tensorflow(valores_entradas_X,valores_a_predecir_Y ,
                                        tf_valores_reales_Y, tf_neuronas_entradas_X, peso, sesgo)
    return neuronas.aprendizaje()


# if __name__ == '__main__':
#     main()
