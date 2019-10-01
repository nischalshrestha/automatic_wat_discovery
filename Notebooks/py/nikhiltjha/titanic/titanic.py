#!/usr/bin/env python
# coding: utf-8

# # Titanic : Machine Learning From Disaster

# ## Predict Survival on the Titanic

# -  Defining Problem Statement
# -  Collecting Data
# -  Explotary Data Analysis
# -  Feature Engineering
# -  Modelling
# -  Testing
# -  Generating Submission File

# ## Defining Problem Statement
# **Analysing what sort of people are likely to survive using feature engineering and various tools of machine learning**

# In[ ]:


from IPython.display import Image
Image(url="data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAkGBxMSEhUQExIVFRUXFRcVFRgVFhcVFRUYFxUXFxUVGBcYHiggGB0lGxgXITEhJSkrLi4uGCAzODMtNygtLisBCgoKDg0OGxAQGC0lHyAtLS0tLSstLS0tLi0tLS0tLS8tLS0tLS0tLS0tLS0tLSstLS0tLS0tLS0wLS0tLS0tL//AABEIALEBHQMBIgACEQEDEQH/xAAbAAACAwEBAQAAAAAAAAAAAAAAAgEDBAUGB//EAEAQAAIBAwIEAwYCBwgBBQEAAAECEQADIRIxBAVBURMiYQYyUnGBkaGxQnKCwdHw8QcUIzNDYqLhsiRTksLSFf/EABkBAQEBAQEBAAAAAAAAAAAAAAABAgMEBf/EACcRAQEAAgICAgAFBQAAAAAAAAABAhEhMQMSQVEiYdHw8QRxgZGh/9oADAMBAAIRAxEAPwD4bUxUVIoAMe9N4h7n71AjO/p/3S0D+K3xH7mjxm+I/c0tRQP4zfEfuakXG+I/c1WKk0DeK3xH7mpF1viP3NV02oxEmJmOk94oG8VviP3NBuN8R+5qupoLDcbMM0DufpUeI0TqP3M0gooHF1viP3NT4jfEfuaStNkUFBuN8R+5qPGb4j9zWw2xFK1sZYeXMqBMb7TMiB1ycfWgy+M3xH7mjxm+I/c0pqKB/Gb4j9zR4rfEfuajH4fjUE0DeK3xH7mjxW+I/c0aRpmczEQdo3nakoH8VviP3NHit8R+5pKkUD+I3xH71Hit8R+5qVSoYUB4rfEfuaPFb4j9zSVNA/it8R+5qPFb4j9zSg12vZr2cu8YXFuBotu51ED3VJxO9Njj+K3xH7mjxW+I/c1FxYJHrUA0DeI3c/c1dwxJnJ6VQDiK0cCN/p++gy1JNAFRQSTQDUlag0EqP6+tRUUUEsIxUVIqKCRThaFA+vzxHaI3+tXhaCkrSFavIoYYoM4oAoNSI9SKAWtiDp/3WIVptPQayuKpmrkMiqblBjujNQqiDkCNhnOdhA+ue1SWIMikoCiimVd8gR885AgfnntQLRRRQFW2rdWW0mSY/L7AVLsBQJcaKpJqWNLQSKttWmcwoJ6nt8z0rs+zPKna9afK+cFcCZEEYP8A39N69Tb9jbwAAhUgt7nvAELOXE5+tB5LheSCdLuNXwghekyXbAEV0eB4dUUXUVtLak1K5z5ZIEnOD8q9Hwfsc2oqLyap0Nm2GBPU+cxGROwmKwe0HK38RNLa/GBuEyIGokHAHl+Xy71LKvDyXNuWeGBdWSjHTn3g0SQe/XPoe1c2fwr2PtTotcMtkZJYRO+PMzR9fxrx1as12zLsGtPBHf6fvrMau4domoqiiiigYt0mlqaigKKKKAoqQaBQNbNaNVZg3T61atBbp61PShTUigy3Eikmr+L6VnoGFOO/9arpmEdQdjjpjag3cO9V8Q1FiYmKq4k0FBqKBRQSRUU2cH8emOn5UtAVIqKlRNA4u0uTS1p4Hg7l1ilq2ztBaFBYgKJY42AA3oM4UnEZr0/J+XWrdu696PEVA1uT5Q2uCvZjGfpjvWz+z32dPFeLdJgJCqTk4BZwM9FA+4HU17rjuXreu2eWolsRb1llEFRkCT3Zj1PWaurpN8vn3D8xZLiuoXHnGpWhtOQIBBAn9LBzjetg54Qt1nuuLmkeEQdz4nmAiJ8pbfAr1t3+zzVEMkkGJLDaQThsYB+1LyfgtN27wKWrD+Hpd3YMSWue4mQeh/OpMavtK8La427d8hLeZtR8zEkAjf65+dd1SlhDcczAzqkzBiJkEdQI/fIublycKzaiDpIJYz5l+cdiflJjNeL9ouateYKJCLsD1Pc/z1rpjj6zdYuXtdRh5lxzXnLt8gOwGwrKKiiudu2zEb52/GoDVFFAUUUUBUigURQANRUipViMgx8qBakioooJq62elUVbbFBoUVZFVo1XMkKT6UHPutJpKmigitHBAas1SRQhgzVg7t9BoMCuFXUTigRjPpVnBcgu3Fe4RACsyjq0AkY7Y/L50zsiYyuNFRTVEVFMzDMSB6mT67RS0RRFAAUAVcLThWYBtE6GI90mQdJIwcgH7UJwjnTCMdRIXB8xG4HeKCP7s+jxdJ0atGqMaonTPePyPaut7P8ACsri4bjW5DCFJDOIhlPZTsZrocp5bbs5veZsMVA1BIB+jNB22/OvS8t5SeIYtaViCToe55QzDMgLMgSB8x61Pyh+dUJzi5Y4fwba27YOsIVnWJ3YdjA39KxWebXhcuXfEYO5QMYzCHUo9B6dZA2qznXA3OHYNd0FiDGkkgRG5MRAmubZtli7kgAe9t2wP67Vn8W2uHTfmvEuS7XNTCYICqFGls4Ag5MEZ27Vq4L2hvedkFtdTHxrpWSxXInVIxqIgACslm+PDX3RbQuPMoOqQBq2ycYmcgGK4nHcUXhVGm2Nl7+p711xxvdYys6h+fc5fiHJZpWflqjqQNvlXnOKbzRXTK1x2M5q51MYAakdP56/hS0VhowaDI/EA/gaWpFRQTQw+tNFTpoEAqSP+6sUUaauhWBRFXi3UFKaRTFSFp9NMq99usbxTQrC1ZboIoimhotCtHFN5M52A+mBWFSRVl25IAn+tXQzaakJTxRFNCvTXQ5DwyXL6W7gkPKjMeaPLt64+tY4qyxeZGVl3Vg4+akEflUs3Fnb6FwPs/ZDCEAkkfjiuqbQS6E2jb0P6Q+W/wCNShDIt1NjDr8mhh+BpuYHU4buB9xj+FeL2uU1Xr1JeHy3nnL/AAOJu2TgK5iMwreZY/ZIrCuOgPz+c/MbdK9j/aLw03LPER76FG/Wt/xVh9FrySrJyQo7kGBj0Fevx32xlebOaysIqFjCrJOwUE/Yb16Plnsu0LcvIxn3LS4d8iAxwEGe8/KqeW2zavtaTzQmokgDKgPqB/RUjy5P6XrFe+ucWRMFQqHAEscBYx0/R77it6c9reTezBvQLmlbaxFtYCjquqPe2nECQcmTWj2stK9hRw0SrAalEsQ1rUAABIxpwB1rfc5zbt6gCdQuFk1CAiwbbSZlydDMNX/uJ2x5zi+YlL1wMzGDc17nU7KysBONOQQvpBmBWblJxO1kt5rzvCcL4barii8ATKeILZY6iIJgkL72AK9UfbgpaIHDLaZQAgW4GQAbYCrGAYx0rJwvL0UJxPET4Z9xQYe6FIlRGUBBMGOnTBLvy+2sm94iM4127VtUJUGSmu4w6htgCY9a6Y8TXyzlzzenjuYcUXPiONbdW6vgDMYHzAHrV9jhgi+I7HSf0ernHQ/KujzG2nCgq6wxhtOJbUAyn5QQa5Z1MTcuyCJCqQIDK0FCrbkHTK9nk7AMmEnNuy576TxDG4NbEBc6FExgx0HXP26CJ59ytzuXx9pIwO5OJMADUegHYVz71arMVk1h4qyBkda2vVPFpIxmMn0mubtrUYdGCZG4EZkzOdogQOvUb5hStWRUEVNMqiKinNLUVaBTRUgU7Ge2ABgAbd4GT6nNaQqrV6gCZE4xmIMjPricetVCnJqgWmZKVFq0jFBlimI9en8ipNRQRFC5MDNMKThn0PHrQaLvCELNZbTTXXuNqUj0ri28NFLwkXUUxJ/dUUEUVNFFfUP7OrovcKLZ3ts1v6El1/8AIj9munzbhSoDdjP8fwrxn9mHMPD4lrROLqY/Wtyw/wCJuV9G51dXT9f+/wB9fO8m8fN/d7ML7YR5T2n5f4/DRIXSyvJmABIYwN/Kxx/CvJcDy21dItradjPvajmehAO+NhO/1r23BseKRrFpSSQEJIhQGkZJ7Lv866Ifh+WqyW1W5xUBLYCFmJlioA3VZOW3Ig5kCvV4ZlJy4eay3h5H2n4N+EZbZlC9pCe7YNvzttICRE4JPU1y149gFZpMgFVM6SuobQZIJUZ/2iNq63tE7KTc4p9fFXASqggeGNTFXYjqCYHTEDAxjs8ufinDjVdc27ZwCSSlpA59cqTPqK6WW1zlki7irrPdMCBoQj/aHthoncnzHfNejVV4dhxF0re4hx4iWwQVRg7amvDEAEjA3kgQfMMN7hrCotm0fFusZ1pqCoASNAUiWLTPfbbauhy3l4slECK99oKKxRrWh7THU5n3xghTtjBkVvHGdzr9/wDGMsvtj5fbuW3LNbLXolSymVhgwYKdxpB37k9KTmXNBwblBD8SSAp1grbYsrK+o4aRIM/F6GcHM+dut1hYvG5cBJN5iQEBbTqYnIywXsJrhNcW1rVlLXJKsGBAcagQbkNqAKEg2xBVkUk7iumVk4jMlvaWdcXbhF12AIBmIIYEdMAEicFWQFZHmqm0CYk9FExsAAomBsBFVDUxLuxJOWZjLNkAnJ8xrSpGBsBOTuTHoOsYHSd+tZi0zDE/zMZrLHWr7rztWXiLgGPvTKtYT5UmWJq2BoZdQDEDPcDp+FZnuEmKpczmsNWqyKiKalNEIaQ05pYqDTFFO29LWhJUgA99vpioWiKBRGi2lF2ns7Ut6ispoBqTUUQUhtMT5VJPoCfyp66vsxdC8QgOzgp9Tlf+QWpldTbWPNZ+F4W8/lCEfMRGYr0PKPZVV/zMswK+iyCP3716fhbIDAx1z9c/vq/jl0XvTB+lebLy3KcO+PjkvL5LctlSVbBUlT8wYP40td/254LwuLYj3birdX9oQ3/JSfrXAr0Y5e0lcMpq6FApkQsdIBJ7ASfsK7HEcsW3YtXxdtuLi3ARubRC+Wf2mn+la2i7l3L7lhrd4IxuqwcLBCgAwVJ6k+nQmvZcv4G/eQXOJci25VPLpDGSB+l+sAW+013bvFcMyJAttqQ3CxK6hER5dOoghbnX9HY9OFzSxdvsdTPa4aADrMO4lm92cZMAEAwBPas5YTe7CZ3qNd/nnhP/AHPgEQsRoZyQLdsqZOThjAee3rtXl+N4tOGlbJNy83+ZfYSZ6hZ3x/OK08XeRVFq0ukaRMRqds+ZiN8k4MjOKXguUMbdziHZEW2A8uYB1Np0oP0iTI+YjepLvn4WyRVyjlFziCpfCqADcfVpWWe5qutmDqZvoBOxNdblvE3Sw4bhPLrJtMwI13cgYYxoXy4HSTnJrHwguOBbDMFZl8uoqpOyyCYGWOTWqzfNq1cUPptuPOSsAlDq8OYxBPcdJrrMNznr99/o55ZaW8LcSwlvwZ8YIHa5lTZYYZEgw26+b598eb5vzPxPJZMLqVWumSSzEkQFknIIlQfxqriuON/yKClkT7vvuNJY5jZVUt5uitGTFci5xbBfDBECQSuFIgo2mAIVl06pksVBJ6DVz+Ikx+accQbepAMSfK51m28aHkQEZiJUypEHuJCC2Y1sfe8wnLPJcavUa0IJJnPWk8I22hlBI3RtQ3XBOkjoQRB6Zxg2kmdb5JznOo+Uw0MGUFTv12HUjDRrW2SYHT17DtPeKc3SekdgPmT8zmd6qGcn0Aj0gflVtpOtXazHaLjaRJrnoSzfWm4y/qONht6+tV2DmstW/EOo94/P84qphViNgj0qs0RXFBFNRFBURTWl3pitW8Mu/wBP31BDVBqTUVpEUUVE1Bos3KLz1Q6sBMUqPIoqSaZxFGnareJGaIopkuFSGUwVII9CDI/GpuCkig+xcGq3UW6vuuiuPkwmPpMfSquYmSrdRg1n/sz4oXOG8MnNpmT9lpuKfvrH7NbObppP1r5kuvJcHu3vGZPM+33C6rFq8P8ATYof1XEj7Moj9avGcFwhuGAQBIBJ6ajAEdSYOK+m8x4XxeGu2viQlZ+JfMv/ACAri+x3LFtOS5Vi6YUgGQNMsvUQW07iZIiM17P6e7x19PP55q7HIOWg29IUtLABMliw1nMR5YUHTKg4ndo93ybkZtab1wIphgFQKlufIXB0gC40gRIxJy2TUci4nh7IvsihgNKyq6TquaQvyUMw3xBmuH7X+1iMEZGPia/FKj3bZZdmA3YHSYyPLntXbrmuPfEeQ5wLtvibltWcAXHRQrMJBuMQgAOcnbqTW7mPLeIsrb8RvO+ptLMWZAsBS0nczt0zVfJuPueI18qrOxnXcmQepEER9Ix6Yrr884W7ae27urPcHiQrSyQcKQdvywakxmt1bedQnLuX2ratc4liGgFbQBFy5qAKk/AuRvkx0waqZXuyzmAFG+EBVSUBgQCQpgAAT6bV8Fw/mm42kke+QWGFhdsxqGmRtnGIq3mPE6OHHjDQz2giW7epLt3Wpa25hdLgEgdCcfOu0w1zf4crlvpdf4y3a8zFxaIXXkKzjBZRgg+dTGDsMTXmuLuNeUu8KirqVDKjOnQ9zTmWkQOupWkqGNczmnM7nENrc4zpUQFWSSdIGBknYdapta3K2wdyqiSFWSdK6mMARMamOAYkCpllvpccdLLvGuym2uFJyoz1VtAAwqBw7KoAjW2TvRbcWxqB/wASQwYe6gi26spBBFwMGUyCB0zkKLwVIQZMMXKw4MKdK5OkAhvMILBjPQBrloKIJlpzBBVRHcSGmRkHEEGZxmRoeAbcFgAwPuGCRpJBDruhkRpYAkGdqN8kycDvgCBnsBAA7DpFK1sznfqDMj5zV9pKbamOzWrdZuLv/oLsN/4U/F8THlXfqe3pWICo1b8QhqUMUEVEUYPSUwFLQFMv3qKkUERVlgb0tW8ON/pRVZFLFWxUFaqKiKXgHAOauK1kfyvP1qVXX4vKGK4/DnMVvXiZEASewzWjl/IXYF28uCUHUkg6Z7CazllJzTGWs1tZFPxNvIo4cZrddsE6YBJ2gCSSegHWto51xcGqYro8XwrWyyOpVgYIO4wCPwqrheHDnTMGDpEe82k6VnpJgZ700PQf2bcw8LijbJxdWP2k8w/46x9a977SEBdRPSZ/fXzXlfLHV0f3Xw6YmIiCBsW+f4nb3LcDcuANcO2fMADBUnzSRE6cD5Dc15PL4Pbye0/y74eWY46rObtxlNtF8xHl/I46R3OM0vEW/At2EJAuYHiRsVHmXu2qfd6kbiK9FwPEcNwy7ydGpiwxB1Aj1MQQBEfWvH8x4y5cY3UH+HbZmLtJVyQAEjBOwwANzOTXbCTHiOeWVy7ci3zi4NaLccWmAAUhNTDy4OkYiJ36DNNyjk9y/dGldzMbBRJGpjsFEjzVs5JyTxXZmIt2lOprlz3LSmTDHYtg46mNproHmLECzwupFOCw9+8TsHAkGJIAA61uY889/X6sW/TpX7lvgrRsIAeLOLjgqy2lIygGRMEiN5g9gOUbYGu9dJLsZaMXCXTUjwRlNszt9JjgwiANGpwwOll1I6wSSTM/1mcVk5rzk8OGRfPeEJqkstkFXU22BGkkiYjbTIrvMfXm9uVu+I08w5meGK22Ae6MWrZMG27OCCekSJgyPMfWvL8ZxHma5dOu40kFTCgEn3cBrL27qdvMJHumXx2+Lw0jUzSGJkhlYZVhMYYKykRBGZwAMhb/ABLrNkYLS1x/K4RhqI1JqTQWk6exiK55Zbbk0quFn1XDnKqSAAoJU6RCgASEP/xNXM2o+HaDaTKgCddxdZdPEAMMwEbCPKO00Q11lVRgErbUsIRWuFghuNAjU8amPWos3tI8ohs+acgShXSIlGBU+YGSHI+cUOFC6QAzEAlpJAkSAsR0MNIOVwe6LRFOtLWscdnRaS/xEYG/5Uly52qnTUatV0xpitBWqwQ1FOVo00ERS1YBQVoEipAptNGmgiKttdaWKkUCi+vf8D/Cjx17/gf4VkqQKm1avHTv+B/hXT5Bwti+7K41ELqX3hgEA9p3FcQW66HIrnh30Y+6Tpb0VhBP0mfpWct2aJdPQ/3FLQJRQNsxWqzkT1x+dWcXZI1IdxI+1ZeEYiPn++vBnvbtjZZtwuaItu6w9dQwdjkfnXuvZ7h+HucPC6hfHmaQfNbkSFnywQdJBzma8t7T8J/l3AN5Q/mPzP2rd7L8YU4iwse8Gtn9pGj/AJBa93iz9sY45zW272mtF1scYbbWw6+GQ6kNbuMWNlmxlCdpjaeoFcrlfBC0y3dydGksD5QygPjoQWj6Gt/KSE4jiLd5zct33IaTqAGom1ckz5l8ufQdq1cTwptko+6YMdZzqXqZAn5Vc89dGOP2t5VzQrabVpW4g8rEHOCUJ2nEY7R61zOfe0xvMIHk0kEsD1BU+UHCgQB2zWtLDXyAMBYLM3uqBkSNiR0+VcTh+IL3A0KxtuQpYSpUSBqzkriPn6CpjcrFskrRx1++4VnTSrrpVdmXBZLgByBPfuR1Brv8q5Pev2S966EtWkNwFhoVhb97QAIMRBbpOxrJzO1eteHxF9NStMK0jIGtAU6Kd46wQetZ7/GcRxJKNcJ8jBU1BUCqpbSq4UYGFG8Adq6Sc8f7Yt45auc8yS4fD4dNFnT5lAkYZSDO+wWSeuoiNUVl4C4wDBEk6Q0gHVb0MG1qR7vYnbPyp1KhVNvUq513W8sB7Wm5ZYjysMEjYkE47cHj+YtcI4axmWIBAAa4TAgE5gwME9TtJnrNYxzu8nQ53zlk1+CwZtbauIUsmrxAQVVTEjDgkLjVB3FecXiQEgDJ1LcBko4j/DuGW/zFLPmIEKRnVMcPxj6PCUmGMCJ1QwKtbGfdeRKxkqvaouWQgKsPOexINsq7o6XEK+8dIODgEbyQOdu+25NJ/u4UTckallVEhyGRjbuSVK6NQE51bxG4Ia4xPlGWPRLaTqfSv6KCdULjJgUcLw2qTIVBGpiMCZMDYM5CsQsgtpParbl6R4dsEL2HvXIzL6QNcGSJHlBiTuYqlLpC6RiZkiQzBtHkYz5llAQDsZNBMYB75GxG2JAI670pI+vUzj0gRioFS1rHFIoa4BuYqm9ejA3rPMmo3a2G6vxCg3U+IVjoNXbDWbq/EKjxV+IVlalpsbDdT4hR4ifEKyE1FNjZ4qfEKPEXuKx0U2Nute4qPFX4hWOimzTZ4q/EKPFX4hWOimzSQK28HYDGMn5CT9qw1ZbeKhXcTgVBUaLnXcQTjpvXf5GlpGYvYe4CpUAgYJBE7iftXkLPExW23xLHv96zXPVe+4fii/8AoITABNwMJgRMhtzE5rYrWRYKNYtC5IIdXgAAGcNMV4W1fciJwOk4qbvMiv6X765bxvwayny9nxPE22TQ4ETMKViR6m3H41l4ThLIYOtvSyupEXEYEAzIVAAAD0mcbV5Ec7uT70/Mn+NP/wD2GO8fdv8A9V0k0z+Lb03M+TJcc3Ve+ggkBeGZlQjBkhgAfpPWsnEcquFjHjMPL/pwzEZxLdu9cxeYqf0EPfNzPzhx+FF7jAx1FVG2BIGPrP40rUuTp80tsVFkMLNse8Dgt08x656Ck4G6vD+dFV3XzKchU6zpK5Mic4xtWReMUf6donuQD+ZihuN7osf7EUH7qPzq+yarTx3HPxjh3dFjCoAwS2ufdAHUk95mk4PlbsY1KR6SJ7b/AMK3cNcsafMGEiJ1BCJ6jUN6vcW512oOQTnVMYmJ09BsAN6e9k4X1324/NeVXSpRXVV38MMSCV8s7yYmNsTWO57PXNGhIllPiYclgDbeGwVUBhIiCZMzFeo4/mJI0+FaKr5Gi1amCfMciYzk0y8aSDkiMxqC7gjbr+VPa1rWnnF9mLiW9S3BqYMHOlxpUqytbHQhg2SRI0rpIzPPHs5cM+YYjoevUYzFess3f8OHKuDqkXCCCCRMjY9oH2pE5pbh202tRxkNIEjA6DrtNT2yjWo85xfJb9xpS35J0qqK0KoLFdWJJGo+ZiTmJOKou8ouKsHSokaupJBeDtKwGiOsT2jt/wB8BBSbYXZcMcCMRAjrse1ZeO5gD1+0gRGxB3p7VdRwF4M6gCcSJYCYHUxue8Cuu/IrY0/+qs+suox3ImR0xHU9s43cHr3x0md4neqiQBGPnH/dXbSrj+DtWyVtxdjJckhOpgCRP3+9c0r1IA9AQAPuSa2uw7D6j/us7NB91T8/61U00W+UO8NbRiI1QQfdjcEhQw9RVFu2skFlU9iASCCMHUYGxq9eecQq6FuuqCIVWIAgyADvg7dqxXL4bpn88zJzVZWPaHmbXbBmQDBLSegWQO/Sl4Xhg8ywWASPKzSe3lGKqdgQMRHbr65mp/vDTIOn9Xyj7CgVljrUEes1GqpViNjQBUjEUtP4p7n70eJ3ANAlFSa28s8L/EFwgE2yFJXUFOpcgblokCI3JnFQYakqexyJHqO4rtcDx9mx4nhy7G2AGuKQjMLusjQpmCAkSRlM7xXK4rinuNqcyc9AN2LHb/czH60FNSKiiqNPD7127fuL9fzoorlmLTtXK4jeiis4dl6VVNRRXZhp4WtlFFZqoO38+lKaKKikf3a6nId6KKvwO1d3+tUjdv1f/stFFYx7by6V/B+qf/I1x36fL95oorqxFbVmu0UVlpmtbfz3pmooqrGW5VZooqqqekooqs0UUUUQUUUUBRRRQFFFFAUUUUH/2Q==")


# ## Collecting Data
# -  Data set is available on kaggle platform and can be downloaded.

# In[ ]:


import pandas as pd
train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")


# ## Explotary Data Analysis

# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train.shape


# In[ ]:


test.shape


# 
# ### Data Discovery
# -  Survived: 0 = No, 1 = Yes
# -  pclass: Ticket class 1 = 1st, 2 = 2nd, 3 = 3rd
# -  sibsp: # of siblings / spouses aboard the Titanic
# -  parch: # of parents / children aboard the Titanic
# -  ticket: Ticket number
# -  cabin: Cabin number
# -  embarked: Port of Embarkation C = Cherbourg, Q = Queenstown, S = Southampton
# 
# **Total rows and columns :**
# We can see that there are 891 rows and 12 columns in our training dataset and 418 rows and 11 column in our testing dataset.
# 

# In[ ]:


train.info()


# We can see Age,Cabin and Embarked has missing value in training dataset.

# In[ ]:


test.info()


# We can see Age,Cabin and Fare has missing value in test dataset.

# In[ ]:


train.isnull().sum()


# Age has 177 missing values, Cabin has 687 missing values and Embarked has 2 missing values in the training dataset.

# In[ ]:


test.isnull().sum()


# Age has 86 missing values ,Fare has 1 missing values and Cabin has 327 missing values in test dataset.

# ### Visualizing Data for Explotary Analysis

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# 
# ### Bar Chart For Categorical Data

# In[ ]:


def bar_chart(feature):
    survived=train[train["Survived"]==1][feature].value_counts()
    dead=train[train["Survived"]==0][feature].value_counts()
    frame=pd.DataFrame([survived,dead])
    frame.index=['Survived','Dead']
    frame.plot(kind='bar',stacked=True,figsize=(10,5))


# Categorical Data:
# -  Pclass
# -  Sex
# -  SibSp
# -  Parch
# -  Cabin
# -  Embarked

# In[ ]:


bar_chart("Sex")


# -  It is observed that there are more female survivor than male

# In[ ]:


bar_chart("Pclass")


# -  This Chart confirms that there are more class 1 survivor and class 3 are more likely to die.

# In[ ]:


bar_chart("SibSp")


# In[ ]:


bar_chart("Parch")


# In[ ]:


bar_chart("Embarked")


# ## Feature Engineering:

# In[ ]:


train_test=[train,test]


# In[ ]:


for dataset in train_test:
    dataset['Title']=dataset['Name'].str.extract(' ([A-Za-z]+)\.',expand=True)


# In[ ]:


train.Title.value_counts()


# In[ ]:


test['Title'].value_counts()


# In[ ]:


title_map={"Mr":0, "Miss":1,"Mrs":2, "Master":3,}


# In[ ]:


train['Title']=train["Title"].map(title_map)
test['Title']=test["Title"].map(title_map)


# In[ ]:


train.Title.isnull().sum()


# In[ ]:


train["Title"].fillna(4,inplace=True)


# In[ ]:


train.Title.isnull().sum()


# In[ ]:


test.Title.isnull().sum()


# In[ ]:


test["Title"].fillna(4,inplace=True)


# In[ ]:


test.Title.isnull().sum()


# In[ ]:


train.head(50)


# In[ ]:


bar_chart("Title")


# In[ ]:


sex_map={"male":0,"female":1}
for dataset in train_test:
    dataset["Sex"]=dataset["Sex"].map(sex_map)


# In[ ]:


bar_chart("Sex")


# ### AGE

# In[ ]:


train["Age"].fillna(train.groupby('Title')["Age"].transform("median"),inplace=True)
test["Age"].fillna(train.groupby('Title')["Age"].transform("median"),inplace=True)


# In[ ]:


train.Age.isnull().sum()


# In[ ]:


test.Age.isnull().sum()


# In[ ]:


train['Age'].value_counts()


# #### Binning the Age
# -  Child:0 :0 to 16
# -  Young:1 :16 to 26
# -  Adult:2 :26 to 36
# -  Mid-Age:3 :36 to 60
# -  Senior:4 :60 and above

# In[ ]:


for dataset in train_test:
    dataset.loc[dataset['Age']<=16, "Age"]= 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 26), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 26) & (dataset['Age'] <= 36), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 36) & (dataset['Age'] <= 60), 'Age'] = 3
    dataset.loc[dataset['Age'] > 60, 'Age'] = 4


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


bar_chart("Age")


# ### Embarked

# In[ ]:


Pclass1=train[train['Pclass']==1]['Embarked'].value_counts()
Pclass2=train[train['Pclass']==2]['Embarked'].value_counts()
Pclass3=train[train['Pclass']==3]['Embarked'].value_counts()


# In[ ]:


df=pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index=["1st Class","2nd Class", "3rd Class"]
df.plot(kind='bar', stacked=True, figsize=(10,5))


# All the 3 class people are majority S embarked.

# In[ ]:


for dataset in train_test:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')


# In[ ]:


embarked_mapping = {"S": 0, "C": 1, "Q": 2}
for dataset in train_test:
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)


# ### Fare

# In[ ]:


train["Fare"].fillna(train.groupby("Pclass")["Fare"].transform("median"), inplace=True)
test["Fare"].fillna(train.groupby("Pclass")["Fare"].transform("median"), inplace=True)


# In[ ]:


train.Fare.value_counts()


# In[ ]:


train.head()

