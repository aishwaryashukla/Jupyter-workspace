import yagmail


class AishMail:
    def __init__(subject, body, to):
        self.subject= subject
        self.body= body
        self.to=to


    def sendmail(self):
        '''
        this function is for sending emails to myself.
        content: is the body of the email.
        subject: is the subject of the email.
        to: reciever of the email, by default its my to me.
        '''

        user = yagmail.SMTP(user='mail.aishwaryashukla@gmail.com', \
                               password='kcqqrerumbpuzlma')
        user.send(to=self.to, \
                     subject=self.subject, \
                     contents=self.body)
