# Proyecto de tratamiento de imagenes por computadora

## Registrado automatico de films escaneados


> Info
 - Proyecto de la materia Timag del IIE de la Univerisdad de la Republica del Uruguay.
 - Programa en Python para detectado automatico de films escaneados.
 
> Proyecto 
 - El proyecto consiste en la deteccion de perforaciones de films mediante uso de matematica y analisis de matrices para posterior deteccion de bordes de los frames mediante transformada de Hough y herramientas de bibliotecas varias (OpenCV, Numpy, Matplotlib, ..). 
 - Recorte de zona encontrada y armado de la pelicula.
 - Algoritmo de backup de ingreso manual en la deteccion del frame luego de orientado mediante perforaciones.
 - Soporta imagenes con distintas orientaciones del film
 
> Instrucciones
  1. Instalacion de bibliotecas necesarias mediante uso de pip:
    - pip install -r requirements.txt
  2. Corriendo el programa
      - Se copia la carpeta con los escaneos filmicos dentro de la carpeta *./imagenes/*.
      - Se corre el programa desde el root del repositorio con:
          - *python -m project*
      - Va a pedir un input con el nombre de la carpeta donde se encuentran las imagenes y luego la cantidad de fps del film a armar.
      - El output se posiciona en subcarpetas dentro de donde se encuentran las imagenes:
          - */aligned/*: se encuentran las imagenes tales que la posicion del film es simpre el mismo
          - */movie/*:   las imagenes recortadas al film
          - */film/*:    se encuentra dos films armados con las imagenes generadas en las dos carpetas anteriores
 3. Si todo sale bien, se muestra en pantalla:
     - Done :::: SUCESS ::::

> Ejemplo de uso:
 - Supongase que se tiene imagenes del film escaneado en la carpeta 'film_x' dentro del directorio './imagenes'. Desde consola situado en el root del repositorio:
     - python -m project
     - Ingrese nombre de la carpeta que contiene el film: film_x
     - Ok
     - Ingrese número de fps: 30
     - Ok
     - Done :::: SUCESS ::::
     
     
