from bottle import route, run
from bottle import get, post, request


a = 8


def calc():
    return 3 * a


@route('/login', method='GET')  # or @get('/login')
def login():
    username = request.query.get('user')
    password = request.query.get('pass')

    # GETで何も渡されていない時はusername,passwordに何も入れない
    username = "" if username is None else username
    password = "" if password is None else password

    return '''
    <form action="/login" method="post">
            Username: <input name="username" type="text" value="{username}"/>
            Password: <input name="password" type="password" value="{password}"/>
            <input value="Login" type="submit" />
        </form>
    '''.format(username=username, password=password)


@route('/login', method='POST')  # or @post('/post')
def do_login():
    # username = request.forms.get('username')
    password = request.json

    return f"username {password}"


def main():
    run(host='localhost', port=8080, debug=True)


if __name__ == '__main__':
    main()
