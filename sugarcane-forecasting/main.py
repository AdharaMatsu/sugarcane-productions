from pruebas import loader_data


def main():
    file_no_weather = 'Mills No Weather'
    file_weather = 'Mills With Weather'

    loader_data(file_no_weather)

    print("Pronóstico realizado con éxito y resultados guardados.")


if __name__ == "__main__":
    main()