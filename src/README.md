# Kurulum
Sözlüğü oluşturabilmek için, <a href="https://book.d2l.ai/install.html">https://book.d2l.ai/install.html</a> adresini kullanarak d2lbook kütüphanesini yüklemeniz gerekir.

# Ayarlar
config.ini dosyasında gerekli ayarlar sunulmuştur. Değişiklik gerekiyorsa, <a href="https://github.com/d2l-ai/d2l-book/blob/master/d2lbook/config_default.ini">here</a> adresindeki örneği takip edebilirsiniz.

# Tema
Temayı değiştirmek için, d2lbook klasörü altındaki "sphinx_template.py" isimli dosyadan html_theme_options seçeneklerini düzenleyebilirsiniz.'primary_color' seçeneği üzerinden ana rengi kırmızı olarak düzenledik.

# Düzenleme  
Sözlük chapters klasörü altındaki sozluk.md dosyasında verilmiştir. Düzenlemek için harfin başlığı altında terimi bulup düzenleme yapabilirsiniz.

# Yeniden Derlemek
## HTML Oluşturmak
build.sh isimli betiği çalıştırdığınızda;

- önceki html dosyalarını kaldırır,
- daha sonra aşağıdaki html oluşturma komutunu çağırır:
```
d2lbook build html
```
- son olarak gömülü metnin çevirisi ve harici bağlantıların yeni sekmede açılması için oluşturulan html dosyalarını düzenleyen revise_html.py betiğini çalıştırır. Bu betiğin çalışabilmesi için "beautifulsoup4" paketinin kurulu olması gerekmektedir.

# yapayogrenmesozlugu.github.io Adresinde Yayın:
Sözlüğü https://github.com/yapayogrenmesozlugu/yapayogrenmesozlugu.github.io adresinde yayınlamak için aşağıdaki komutu kullanın:
```
d2lbook deploy html
```
