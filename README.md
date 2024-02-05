# NLQxform-UI

The code for our paper **NLQxform-UI: A Natural Language Interface for Querying DBLP Interactively** (currently under review).


## Built With

* [![Flask][Flask.js]][Flask-url]

* [![Bootstrap][Bootstrap.com]][Bootstrap-url]

  

## Getting Started

### Installation

```bash
git clone https://github.com/ruijie-wang-uzh/NLQxform-UI.git
```
Please download our fine-tuned BART model from [The Open Science Framework (OSF)](https://osf.io/tzkfd/?view_only=0a0e8eb8999440688f4e915f7309e1df) and move it to the directory `./src/ckpt/`

### Prerequisites

- Make sure your Python environment includes libraries shown as below:

  ```powershell
  transformers, torch, beautifulsoup4, SPARQLWrapper, flask, pandas, pydantic
  ```

- Or you can create a new virtual environment and install them:

  ```bash
  conda create -n [YOUR_ENVIRONMENT_NAME] python=3.10
  conda activate [YOUR_ENVIRONMENT_NAME]
  
  pip install transformers
  pip install torch 
  pip install beautifulsoup4
  pip install SPARQLWrapper
  pip install flask
  pip install pandas
  pip install pydantic
  ```

### Deployment

```shell
cd NLQxform-UI

flask run --host 0.0.0.0 --port 8087 --debug
#or you can define the host address and port number in app.py:
#	app.run('0.0.0.0', port=8087, debug=True)
```

`--host`: 

  The default is the localhost address. If you set it to 0.0.0.0, the server would listen on all available network interfaces.

`--port`: 

  The default is 5000. You can set another port number that is available on your device.

Upon running, a `logs/` folder will be automatically created, which includes log records named by the date.

The log records are updated in real time while the system is running. You can use them to monitor the system state and also check system configurations, e.g., the IP of the server.

If running on your local machine, you can directly access the system at [http://127.0.0.1:8087](http://127.0.0.1:8087).

If running on a remote server, you can find the address to use from the log records.


[Flask.js]: https://img.shields.io/badge/Flask-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[Flask-url]: https://flask.palletsprojects.com/en/3.0.x/
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
