## Snowmapper Tajikistan
This repository contains the code for the snowmapper tajikistan dashboard. The dashboard is a web application that displays snow cover data for Tajikistan. The data is sourced from the [snowmapperForecast](https://github.com/joelfiddes/snowmapperForecast) model implemented by @joelfiddes.

## Architecture
This project comes in 2 components: A data processor and a web interface. The data processor is a python script that downloads the latest snow data from the TopoPyScale and FSM model and stores it in local files optimized for display. The web interface is a web application that displays the snow-related data on a map. Both components are run in a docker container.

### Data Processor
The data processor is a python script that downloads the latest snow data from the TopoPyScale and FSM model and stores it in local files optimized for display. The data processor is run as a cron job that runs every day at 00:00 UTC. It further removes the old data and keeps only the latest data on the server. Data processing can take several minutes to hours, requires a stable internet connection, and a server with sufficient memory resources (we recommend at least 8 GB RAM).

### Web Interface
The web interface is a web application that displays the snow-related data on a map. The web interface is built using the Holoviews and Bokeh libraries and the panel dashboard library. It is run as a web server that serves the web application. The web interface is run as a docker container that is started when the server is started.

## Instructions
### Requirements
- Ubuntu 20.04 LTS
- Storage: 50 GB
- Memory: 8 GB
- Docker engine (Docker version 27.1.2 or higher) [Installation instructions](https://docs.docker.com/engine/install/ubuntu/)
- git (git version 243 or higher) (Installation: `sudo apt-get install git`)

Note: The installation of Docker Engine alone requires 2 GB, each docker image will require close to 2 GB and the caching of netCDF files will require another 2 GB of free storage.

### Deployment steps
#### Clone GitHub repository and adapt the environment variables
Clone the GitHub repository
```bash
git clone <repo-url>
```

Change to the repository directory
```bash
cd snowmapper-tajikistan
```

Edit the `.env` file and set the environment variables.
```bash
vi .env
```

Copy the .pem file to the server running the snowmapperForecast model operationally and set the path to the .pem file in the .env file relative to the folder /app/processing.

#### Pull and test-run the docker containers
The docker image are hosted on DockerHub. We have prepared a bash script that pulls the processing docker image and runs it in a container.
```bash
bash run_data_processor.sh
```

Check the docker logs to see if the data processor ran successfully
```bash
docker logs <container-id>
```

To test-run the web interface, run the following docker compose command
```bash
docker compose up -d
```

#### Operationalize the web interface
Run the docker container with the web interface
```bash
docker run --rm --env-file .env -p 5006:5006 snowmapper-tajikistan
```

Optionally set up a reverse proxy to serve the web interface on port 80 ([Instructions](https://www.docker.com/blog/how-to-use-the-official-nginx-docker-image/)).

#### Watchtower
Optionally, you can use the watchtower to automatically update the docker container when a new image is available.
```bash
docker run -d --name watchtower -v /var/run/docker.sock:/var/run/docker.sock containrrr/watchtower
```


#### Operationalize the data processor
Define regular cron jobs
```bash
crontab -e
```

Add the following line to the crontab file to periodically run the data processor at 1:00 UTC
```bash
1 0 * * * bash ~/taj_snowmapper_dashboard/run_data_processor.sh >> ~/taj_snowmapper_dashboard/logs/crontab_processor.log 2>&1
```

And add the following line to the crontab file to periodically restart the web interface at 2:00 UTC
```bash
1 1 * * * docker restart taj-snowmapper-dashboard >> ~/taj_snowmapper_dashboard/logs/crontab_dashboard.log 2>&1
```


