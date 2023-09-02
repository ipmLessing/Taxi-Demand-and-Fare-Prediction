![](https://github.sydney.edu.au/kcho4755/cs25-3/blob/master/web_app/banner.png)

# Welcome to the Web Application 

This is the web application directory. All the stuff including front-end, backend, testing, and deployment is included. The zip file is the entire github repo with detailed commits over the entire lifecycle. The unziped verison is not included due to the size limitations. 

## Structure of the Web Application

In this 'cs25_py36' directory:

### Web application Project Root

The main root for the entires for the settings and routing is located in the **project_root** folder.

For the settings, pls go to the sub-dir settings which including three settings files name ended with *.py*.
* base.py <- this is the file for the general settings, ***ALL** the important settings are here.
* dev.py <- this is the file settings in the local development environment. The based.py is included so that if you change the base.py it affacts the dev.py env as well.
* production.py <- this is the file settings for the deployment in AWS environment. ALL security stuff (HTTPS), domain name, IP routing, and Database configuration is here.

### Static files 

All the static files are located in the subfolder *static* for the HTML, JS, images. Once you add the files, pls input the command **./manage.py collectstatic --noinput** to update the static file list cache in Django Framework (Special stuff and sometimes if you don't run it, the static files will not be updated in cache and you will get the old file when you keep flashing you html page). For more details pls go to: https://docs.djangoproject.com/en/3.2/howto/static-files/

### Template files

All the pages and template are located in the location of *cs25_py36 > project_root > templates*.

inside this folder, three folder corresponding to the main index (project_root), demand predition page (maps), and fare prediction page (faremap) is created separately. Pages are extended from the root page named *horizontal_index.html*. The data upload page is standalone named *trip_upload.html*. Manual: https://docs.djangoproject.com/en/3.2/topics/templates/

### Routing

The main routings from root to other page (called *app* in Django) are located in the location of *cs25_py36 > project_root > urls.py*. With the paths, the request will be forwarded to the separated *app*: 
* demand prediction (folder maps)
* fare prediciton (folder faremap)
* upload page (folder csvfile)

The detail routes are located in their own directly file named *urls.py*; e.g. *cs25_py36 faremap > urls.py*. The structure is like request first to check the path in s25_py36 > project_root > urls.py and it routes the request to the app's path faremap > urls.py. For more details: https://docs.djangoproject.com/en/3.2/topics/http/urls/

### Database

Database is not included in this zip file but the creation and tables are defined for the rebuilding based on different local development env on developers. To rebuild the database. Details: https://docs.djangoproject.com/en/3.2/ref/databases/

Create the initial migrations and generate the database schema commands:

```
python manage.py makemigrations
python manage.py migrate
```
For more operations, pls check this article: https://simpleisbetterthancomplex.com/tutorial/2016/07/26/how-to-reset-migrations.html

### Requirements

All the necessary packages are pip freezed to the *requirenment.txt* file in *cs25_py36 > requirement.txt*. 

***We using Python 3.6.8 for the web application*** Since most of the framework are supported and the AWS also supported this version better than the latest versions.


## Running the Application

### Install the environment

To get the environement ready, please first rebuild environment by downloading the necessary packages: 

For user who want to install the environment with python, please do followings:
1. install the conda or eb-virt 
2. create the env with the conda or eb-virt

For example with python3.6 and conda:
```
$ conda create -n env python==3.6
$ conda activate env
$ pip install -r requirements.txt
```

This virtual env will be installed with all the package that specificed in the *requirements.txt* in order to run the program

### Register Google Maps APIs

Please go to the google maps web api (https://developers.google.com/maps) to register the account with $200 US free tier every month. Then regist the following APIs:

![](https://github.sydney.edu.au/kcho4755/cs25-3/blob/master/web_app/google_apis.png)

After all the above APIs are registered. Pls chnage the GOOGLE_API_KEY in the settings file which located in: *cs25_py36/project_root/settings/based.py* Line 162. in my case is 

```
# Google API KEY
GOOGLE_API_KEY = 'AIzaSyAb-ZP9gC2qJni434bZmI8Ii-fvNt-xd30'
```


### Run the local server

```
./manage.py runserver

or 

python manage.py runserver
```

By default it runs the dev environment (dev.py, but you can change it if you want e.g. connect to the prodcution AWS env/database), with the command *./manage.py runserver* in the cs25_py36 location which has the file *manage.py* (A virtual web server script).

### Check the pages are ok 

To check if the server is doing good, pls go and open the browser with urls:
1. demand page: http://127.0.0.1:8000/demand
2. fare page: http://127.0.0.1:8000/fare
3. data upload: http://127.0.0.1:8000/upload-csv


## Deployment

Please checkout the Final Report for the details, here just the tech. details. 

### Please get an AWS Account first

Pls go: https://aws.amazon.com/

### Steps
1.	install EB CLI command line (if not installed in the requirements.txt)
2.	cd into the application directory in artifacts with the path of: ‘~/CS25-3/cs25_py36’
3.	EB environment setup with the command: eb init -i 
4.	input the following information in the command prompt:
4.1.	account api key (download from AWS account console)
4.2.	availability zones as: 8) ap-southeast-2 : Asia Pacific (Sydney)
4.3.	create new application with a name (e.g, NYCTAXI)
4.4.	select the platform exactly as: 3) Python 3.6 running on 64bit Amazon Linux
4.5.	setup SSH for EC2 install: yes
5.	deploy the application by command: eb deploy 

### AWS EC2 env config

located in *cs25_py36/.ebextensions/django.config* for all the init setting for the EC2 machine.

### AWS RDS

pls go: https://aws.amazon.com/rds/ and open the PostgreSQL database. Version doesn't matter, be careful the ***COST***.

#### To link the DB to Django: 

*settings in cs25_py36/project_root/settings/production.py* this is set to auto adapted to EB environment DB. When you config the DB in the EB web console, you don't need to change anything below and it will work. On the other hand, If you have another VPS and you want to connect the DB directly without the EB, you can change the settings to your DB settings:

```
#line 41 - 50
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': os.environ['RDS_DB_NAME'],
        'USER': os.environ['RDS_USERNAME'],
        'PASSWORD': os.environ['RDS_PASSWORD'],
        'HOST': os.environ['RDS_HOSTNAME'],
        'PORT': os.environ['RDS_PORT'],
    }
}
```

### AWS Domain Name

pls go: https://aws.amazon.com/route53/ and register the domain name, follow the offical instruction link the domain to the Elastic Beanstalk you've created in the previous steps. **Please remember to update the settings in cs25_py36/project_root/settings/production.py** Line 14-16. 

In my case:

```
# Set the ip manually if necessary
ALLOWED_HOSTS += [
    'kasengchou.com',
    'cs25-3.kasengchou.com',
    'cs25-3.ap-southeast-2.elasticbeanstalk.com',
]

...

# Also the BASE_URL in line 30: 
BASE_URL = 'cs25-3.kasengchou.com'

```


kasengchou.com and cs25-3.kasengchou.com is the domain and subdomain allowed to access when request passing through the Django Firewall. The cs25-3.ap-southeast-2.elasticbeanstalk.com is the domain that given by Elastic Beanstalk when creating the EB App. Please change to your doamin to link the allow these domain to access this app. 

*This is just for Django Firmwork, you still need to link the domain to this EB EC2 Machine in AWS Route 53*

### HTTPS, Load Balancer, and Debugging

These components are located in the Elastic Beanstalk web console, for the details pls check the Final Report on how to config it. 

HTTPS enabling (in *cs25_py36/project_root/settings/production.py*): 

```
# line 10
# HTTPS REDIRECT
SECURE_SSL_REDIRECT = True

...

# line 33 - 38
# Redirect of http to https:
SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')
SECURE_HSTS_SECONDS = 60
SECURE_CONTENT_TYPE_NOSNIFF = True
SECURE_BROWSER_XSS_FILTER = True
SECURE_SSL_REDIRECT = True
SESSION_COOKIE_SECURE = True

```

### Update the application in AWS after code changing

use command: *eb deploy*

### Checking the application status

1. use command: *eb status*
2. SMS/Email Alert
3. Go to the cloud watch console for detailed CPU loads and traffic


## Futher Development or Debug for local env

To develop the function with TDD, e.g. the faremap *app* for the fare prediction. Write the new function in file: *cs25_py36 > faremap > tests.py* Then run it with command *./manage.py test faremap*. Necessary functions are tested with clear error msg. The tests.py is located in each *app*. You can also create one if not exist to test anything you want. Checkout: https://docs.djangoproject.com/en/3.2/topics/testing/overview/

