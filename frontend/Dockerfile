FROM nginx:alpine

# Copy static frontend web assets into the default Nginx html serving folder
COPY . /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
