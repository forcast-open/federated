# Install Forcast Federated Learning

There are a few ways to set up your environment to use Forcast Federated Learning (FFL):

*   The easiest way to learn and use FFL requires no installation; run the
    FFL tutorials directly in your browser using
    [Google Colaboratory]().
*   To use FFL on a local machine,
    [install the FFL package](#install-ffl-using-pip) with
    Python's `pip` package manager.
*   If you have a unique machine configuration,
    [run locally de FFL library](#run-the-ffl-library-locally-from-github) from
    source.

## Install FFL using `pip`

### 1. Install the Python development environment.

On Ubuntu:

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">sudo apt update</code>
<code class="devsite-terminal">sudo apt install python3-dev python3-pip  # Python 3</code>
<code class="devsite-terminal">sudo pip3 install --user --upgrade virtualenv</code>
</pre>

### 2. Create a virtual environment.

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">virtualenv --python python3 "venv"</code>
<code class="devsite-terminal">source "venv/bin/activate"</code>
<code class="devsite-terminal tfo-terminal-venv">pip install --upgrade pip</code>
</pre>

Note: To exit the virtual environment, run `deactivate`.

### 3. Install the FFL Python package.

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal tfo-terminal-venv">pip install --upgrade forcast_federated_learning</code>
</pre>

### 4. Test FFL.

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal tfo-terminal-venv">python -c "import forcast_federated_learning as ffl; print(ffl.FederatedModel())"</code>
</pre>

Success: The latest FFL Python package is now installed.

## Run the FFL library locally from github

Runing the library locally is helpful when you want to:

*   Make changes to FFL and test those changes in a component
    that uses FFL before those changes are submitted or
    released.
*   Use changes that have been submitted to FFL but have not been released.

### 1. Install the Python development environment.

On Ubuntu:

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">sudo apt update</code>
<code class="devsite-terminal">sudo apt install python3-dev python3-pip  # Python 3</code>
<code class="devsite-terminal">sudo pip3 install --user --upgrade virtualenv</code>
</pre>

### 2. Clone the FFL repository.

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">git clone https://github.com/forcast-open/federated.git</code>
<code class="devsite-terminal">cd "federated"</code>
</pre>

### 3. Create a virtual environment.

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">virtualenv --python python3 "venv"</code>
<code class="devsite-terminal">source "venv/bin/activate"</code>
<code class="devsite-terminal tfo-terminal-venv">pip install --upgrade pip</code>
</pre>

Note: To exit the virtual environment, run `deactivate`.

### 4. Install the FFL requirements.

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal tfo-terminal-venv">pip install -r requirements.txt</code>
</pre>

### 5. Test FFL.

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal tfo-terminal-venv">python -c "import forcast_federated_learning as ffl; print(ffl.FederatedModel())"</code>
</pre>

Success: The latest FFL Python package is now installed.