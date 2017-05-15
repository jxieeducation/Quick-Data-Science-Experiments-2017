### Link
* https://drive.google.com/drive/folders/0B0b2FfTVAWAMTDZOWnN3VDhUOWc

### Docker
```
docker run --hostname=quickstart.cloudera \
           --privileged=true \
           --name=cloudera \
           -t -i -d \
           -p 8889:8889 \
           -p 8887:8888 \
           -p 7180:7180 \
           -p 8088:8088 \
           -p 8042:8042 \
           -p 10020:10020 \
           -p 19888:19888 \
           -v /Users/jasonxie:/media/notebooks \
           ankittharwani/mids-cloudera-hadoop:latest \
           bash -c '/root/startup.sh; /usr/bin/docker-quickstart'
```
