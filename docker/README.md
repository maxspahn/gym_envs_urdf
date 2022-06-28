# Creating the docker image

In here you find the docker file to create the docker image that allows 
you to run the examples inside a docker.

```bash
  docker build -t urdf_gym_docker .
```

Creating the container will take a while.

Then you can run the example using 
```bash
xhost + # enables the docker to access the screen
docker run --rm --env DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix:ro urdf_gym_docker
```

If you want to enter the docker and run the examples inside the docker you would call
```bash
docker run --rm --env DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix:ro -it urdf_gym_docker bash
```
