// app.js

const app = document.getElementById('app');

// Site content
const content = `
    <header id="header">
        <div class="container">
            <nav>
                <ul>
                    <li><a href="#home">Accueil</a></li>
                    <li><a href="#services">Services</a></li>
                    <li><a href="#benefits">Avantages</a></li>
                    <li><a href="#contact">Contact</a></li>
                </ul>
            </nav>
        </div>
    </header>

    <section id="home" class="hero">
        <div class="container">
            <div class="hero-content">
                <h1>Sahara Analytics</h1>
                <p>Révolutionnez Votre Gestion du Risque de Crédit</p>
                <a href="#contact" class="btn">Demander une Démo</a>
            </div>
        </div>
    </section>

    <section id="services" class="services">
        <div class="container">
            <h2>Nos Services</h2>
            <div class="services-grid">
                <div class="service">
                    <i class="fas fa-chart-line"></i>
                    <h3>Analyse Automatisée du Risque</h3>
                    <p>Évaluez la solvabilité des clients avec précision grâce à nos modèles de Machine Learning.</p>
                </div>
                <div class="service">
                    <i class="fas fa-shield-alt"></i>
                    <h3>Détection de la Fraude</h3>
                    <p>Protégez votre institution contre les activités frauduleuses en temps réel.</p>
                </div>
                <div class="service">
                    <i class="fas fa-chart-pie"></i>
                    <h3>Gestion de Portefeuille</h3>
                    <p>Optimisez vos stratégies de prêt et améliorez la rentabilité.</p>
                </div>
            </div>
        </div>
    </section>

    <section id="benefits" class="benefits">
        <div class="container">
            <h2>Les Avantages Sahara</h2>
            <div class="benefits-grid">
                <div class="benefit">
                    <i class="fas fa-coins"></i>
                    <h3>Réduction des Pertes</h3>
                    <p>Minimisez les pertes sur créances en identifiant proactivement les risques.</p>
                </div>
                <div class="benefit">
                    <i class="fas fa-tachometer-alt"></i>
                    <h3>Efficacité Opérationnelle</h3>
                    <p>Automatisez les tâches manuelles pour une meilleure productivité.</p>
                </div>
                <div class="benefit">
                    <i class="fas fa-lock"></i>
                    <h3>Conformité Renforcée</h3>
                    <p>Restez en conformité avec les réglementations en constante évolution.</p>
                </div>
            </div>
        </div>
    </section>

    <section id="testimonials" class="testimonials">
        <div class="container">
            <h2>Témoignages</h2>
            <!-- Add testimonials content here -->
        </div>
    </section>

    <section id="faq" class="faq">
        <div class="container">
            <h2>Questions Fréquentes</h2>
            <!-- Add FAQ content here -->
        </div>
    </section>

    <section id="contact" class="cta">
        <div class="container">
            <h2>Prêt à Transformer Votre Gestion du Risque de Crédit ?</h2>
            <p>Contactez-nous dès aujourd'hui pour une démonstration personnalisée.</p>
            <form id="contact-form">
                <input type="email" id="email" placeholder="Votre adresse e-mail" required>
                <button type="submit" class="btn">Demander une Démo</button>
            </form>
        </div>
    </section>

    <footer>
        <div class="container">
            <p>&copy; 2024 Sahara Analytics. Tous Droits Réservés.</p>
        </div>
    </footer>
`;

// Render content
app.innerHTML = content;

// Sticky header
window.addEventListener('scroll', function() {
    const header = document.getElementById('header');
    if (window.scrollY > 50) {
        header.classList.add('scrolled');
    } else {
        header.classList.remove('scrolled');
    }
});

// Smooth scrolling
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        document.querySelector(this.getAttribute('href')).scrollIntoView({
            behavior: 'smooth'
        });
    });
});

// Form validation
document.getElementById('contact-form').addEventListener('submit', function(e) {
    e.preventDefault();
    const email = document.getElementById('email').value;
    if (email) {
        alert('Merci pour votre intérêt ! Nous vous contacterons bientôt.');
        this.reset();
    } else {
        alert('Veuillez entrer une adresse e-mail valide.');
    }
});