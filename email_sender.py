from email.message import EmailMessage
from bs4 import BeautifulSoup
from datetime import datetime
import smtplib
import json
import re


jsonData=json.load(open("email_config.json","r",encoding="utf-8"))

def constructhtml(data,Class="N/A",Phase="N/A",Filename="N/A",colunmIndex="N/A",message="N/A",Count="N/A"):
    now=datetime.now()
    if Class=="SUCCESS":
        newBlock='<p class="note"><b>Frigginyeah data anomaly detection,</b>Below mentioned file(s) are completed successfully.</p>'
        noteTemplate=BeautifulSoup(newBlock,'html.parser')
        data.body.find('p',attrs={'class':'note'}).replace_with(noteTemplate)
        #input(data.prettify())
    tableBlock=data.find('body').find('table')
    tableBlock.find('td',attrs={"name":"date"}).string=now.strftime("%Y/%m/%d")
    tableBlock.find('td',attrs={"name":"time"}).string=now.strftime("%H:%M:%S")
    tableBlock.find('td',attrs={"name":"class"}).string=Class
    tableBlock.find('td',attrs={"name":"phase"}).string=Phase
    #input(Filename)
    tableBlock.find('td',attrs={"name":"filename"}).string=Filename
    tableBlock.find('td',attrs={"name":"count"}).string=str(Count)
    tableBlock.find('td',attrs={"name":"columnindex"}).string=colunmIndex
    tableBlock.find('td',attrs={"name":"message"}).string=message
    #input(data.prettify())
    return data.prettify()
                    
def send_mail(Class="N/A",Phase="N/A",Filename="N/A",colunmIndex="N/A",message="N/A",count="N/A"):
    # construct html
    htmlTemplate=BeautifulSoup(open("email_template.html","r",encoding="utf-8").read(),'lxml')
    mailcontent=constructhtml(htmlTemplate,Class,Phase,Filename,colunmIndex,message,count)
    # construct email
    email = EmailMessage()
    if Class !="SUCCESS":
        email['Subject'] = re.sub("^NOTIFICATION","CRITICAL",jsonData['subject'],flags=re.I|re.M)
    else:
         email['Subject'] = jsonData['subject']
    email['From'] = jsonData['from']
    email['To'] = jsonData['to']
    email.set_content(mailcontent, subtype='html')
    server = smtplib.SMTP(jsonData['smtp_server'])
    server.starttls()
    server.login(jsonData['from'], jsonData['password'])
    server.send_message(email)
#send_mail("N/A","N/A","N/A","N/A","N/A",123)
