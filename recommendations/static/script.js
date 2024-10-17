document.addEventListener('DOMContentLoaded', (event) => {
    const movieCards = document.querySelectorAll('.movie-card');
    
    movieCards.forEach(card => {
        card.addEventListener('mouseenter', () => {
            card.style.boxShadow = '0 8px 16px rgba(229, 9, 20, 0.6)';
        });
        
        card.addEventListener('mouseleave', () => {
            card.style.boxShadow = 'none';
        });
    });
});
