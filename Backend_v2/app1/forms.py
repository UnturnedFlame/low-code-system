from django import forms
from django.contrib.auth.models import User
from app1.models import CaptchaModel


class RegistrationForm(forms.Form):
    username = forms.CharField(max_length=20, min_length=2, error_messages={
        'required': 'Username is required',
        'max_length': 'Username has to be at least 2 characters',
        'min_length': 'Username has to be at most 20 characters'
    })
    email = forms.EmailField(error_messages={'required': 'Email is required', 'invalid': 'Email is invalid'})
    captcha = forms.CharField(max_length=4, min_length=4)
    password = forms.CharField(max_length=20, min_length=8)

    def clean_email(self):
        email = self.cleaned_data.get('email')
        exists = User.objects.filter(email=email).exists()
        if exists:
            raise forms.ValidationError('the email is already registered')
        return email

    def clean_captcha(self):
        captcha = self.cleaned_data.get('captcha')
        email = self.cleaned_data.get('email')

        captcha_model = CaptchaModel.objects.filter(email=email, captcha=captcha).first()
        if not captcha_model:
            raise forms.ValidationError('email and captcha do not match')
        captcha_model.delete()
        return captcha
