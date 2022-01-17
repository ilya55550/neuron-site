from django.db import models
from django.contrib.auth.models import User, AbstractUser
from django.db.models.signals import post_save
from django.dispatch import receiver


class CustomUser(AbstractUser):
    researcher = models.BooleanField(default=False, blank=True)

# class ProfileUser(models.Model):
#     user = models.OneToOneField(User, on_delete=models.CASCADE)
#     researcher = models.BooleanField(default=False)
#
#     @receiver(post_save, sender=User)
#     def create_user_profile(sender, instance, created, **kwargs):
#         if created:
#             ProfileUser.objects.create(user=instance)
#
#     @receiver(post_save, sender=User)
#     def save_user_profile(sender, instance, **kwargs):
#         instance.profile.save()

