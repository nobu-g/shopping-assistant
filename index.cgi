import sys
import io
import os
import cgi
import zenhan
import xmlrpc.client as xmlrpc_client
from masked_lm_conf import host, port, DEFAULT_SENTENCE


def print_header():
    print("<html>")
    print("<head>")
    print("<title>BERT Masked LM</title>")
    print("<link rel='stylesheet' href='style.css'>")
    print('<!--')
    print('table {border-left: solid 1px #999999; border-bottom: solid 1px #999999; }') 
    print('td {font-size: 11pt;}')
    print('td, th { border-top: 1px solid #999999; border-right: 1px solid #999999;}')
    print('pre {font-family: "ＭＳ ゴシック","Osaka-等幅","さざなみゴシック","Sazanami Gothic",DotumChe,GulimChe,BatangChe,MingLiU, NSimSun, Terminal; white-space:pre;}')
    print('-->\n')
    print('</style>\n')
    print('<meta HTTP-EQUIV="content-type" CONTENT="text/html"; charset="utf-8">')
    print('</head>')
    print('<body>')
    print('<script src="https://d3js.org/d3.v4.min.js"></script>')    
    print("<h2><a href='index.cgi'>BERT Masked LM</a></h2>")


def print_footer():
    print('</body>')
    print('</html>')


def print_form(sentence):
    print('<form>')
    print('<input name="sentence" size="120" value="{}" />'.format((sentence if sentence is not None else "")))
    print('<input type="submit" value="解析">')
    print('</form>')


def main():
    print("Content-Type: text/html")
    print()

    f = cgi.FieldStorage()
    sentence = f.getfirst('sentence', DEFAULT_SENTENCE)
        
    if sentence is not None:
        sentence = zenhan.h2z(sentence)
        
    print_header()    
    print_form(sentence)

    masked_lm_client = xmlrpc_client.ServerProxy('http://{}:{}'.format(host, port))
    prediction = masked_lm_client.get_predictions(sentence)
    pid = os.getpid()
    filename = "json/prediction_{}.json".format(pid)
    with open(filename, mode='w', encoding='utf-8') as outfile:
        outfile.write(prediction)
        
    json_string = """
d3.json("{}", function (error,data) {{

    function tabulate(data, columns) {{
    var table = d3.select('body').append('table')
    var thead = table.append('thead')
    var tbody = table.append('tbody');

    // append the header row
    thead.append('tr')
        .selectAll('th')
        .data(columns).enter()
        .append('th')
        .text(function (column) {{ return column; }});

    // create a row for each object in the data
    var rows = tbody.selectAll('tr')
        .data(data)
        .enter()
        .append('tr');

    // create a cell in each row for each column
    var cells = rows.selectAll('td')
        .data(function (row) {{
        return columns.map(function (column) {{
            return {{column: column, value: row[column]}};
        }});
        }})
        .enter()
        .append('td')
        .html(function (d) {{ return d.value; }});

        cells.filter(function(d, i) {{ return i == 0}})
        .attr("style", "background-color:aliceblue");

        cells.filter(function(d, i) {{ return i == 1}})
        .attr("style", "background-color:lavenderblush");
        
    return table;
    }}

    // render the table(s)
    tabulate(data, ['input', 'predictions']); // 2 column table

}});
""".format(filename)[1:-1]
        
    print("<script>")
    print(json_string)
    print("</script>")
        
    print_footer()


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    main()

