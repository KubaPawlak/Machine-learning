from task_1.tmdb.client import Client

def main()-> None:
    client = Client()
    movie = client.get_movie(62)
    print(movie)

if __name__ == '__main__':
    main()



