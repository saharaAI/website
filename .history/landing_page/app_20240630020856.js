// Sticky header
const header = document.getElementById('header');
window.addEventListener('scroll', () => {
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
const contactForm = document.getElementById('contact-form');
contactForm.addEventListener('submit', function(e) {
    e.preventDefault();
    const name = document.getElementById('name').value;
    const bank = document.getElementById('bank').value;
    const email = document.getElementById('email').value;
    const phone = document.getElementById('phone').value;

    if (name && bank && email) {
        // Here you would typically send this data to a server
        console.log('Form submitted:', { name, bank, email, phone });
        alert('Merci pour votre intérêt ! Nous vous contacterons bientôt.');
        this.reset();
    } else {
        alert('Veuillez remplir tous les champs obligatoires.');
    }
});

// GSAP animations
gsap.registerPlugin(ScrollTrigger);

gsap.from('.hero-content', {
    duration: 1,
    y: 50,
    opacity: 0,
    ease: 'power3.out'
});

gsap.utils.toArray('.service, .benefit, .case-study').forEach(element => {
    gsap.from(element, {
        scrollTrigger: {
            trigger: element,
            start: 'top 80%',
        },
        y: 50,
        opacity: 0,
        duration: 1,
        ease: 'power3.out'
    });
});

// Initialize animations
document.addEventListener('DOMContentLoaded', () => {
    // Any additional initialization if needed
});
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
    const name = document.getElementById('name').value;
    const bank = document.getElementById('bank').value;
    const email = document.getElementById('email').value;
    const phone = document.getElementById('phone').value;

    if (name && bank && email) {
        // Here you would typically send this data to a server
        console.log('Form submitted:', { name, bank, email, phone });
        alert('Merci pour votre intérêt ! Nous vous contacterons bientôt.');
        this.reset();
    } else {
        alert('Veuillez remplir tous les champs obligatoires.');
    }
});

// Animate elements on scroll
const animateOnScroll = () => {
    const elements = document.querySelectorAll('.service, .benefit, .case-study');
    elements.forEach(element => {
        const elementTop = element.getBoundingClientRect().top;
        const windowHeight = window.innerHeight;
        if (elementTop < windowHeight - 50) {
            element.style.opacity = '1';
            element.style.transform = 'translateY(0)';
        }
    });
};

window.addEventListener('scroll', animateOnScroll);

// Initialize animations
document.addEventListener('DOMContentLoaded', () => {
    animateOnScroll();
});