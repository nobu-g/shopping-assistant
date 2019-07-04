import urllib.request, json
import requests
import pprint


url = "http://lotus.kuee.kyoto-u.ac.jp/~ueda/shopping-assistant/index.cgi/post"
obj = {"utterance" : "あいうえお"}


def main():
    method = "POST"
    headers = {"Content-Type" : "application/json"}

    # PythonオブジェクトをJSONに変換する
    json_data = json.dumps(obj).encode("utf-8")

    # httpリクエストを準備してPOST
    request = urllib.request.Request(url, data=json_data, method=method, headers=headers)
    with urllib.request.urlopen(request) as response:
        response_body = response.read().decode("utf-8")
        print(response_body)


def main2():
    response = requests.post(
        url,
        json.dumps(obj),
        headers={'Content-Type': 'application/json'}
    )
    pprint.pprint(response.body())


if __name__ == '__main__':
    main()
