(function () {
  function startAnimation() {
    try {
      const canvas = document.getElementById('network-canvas');
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      let width;
      let height;
      let particles = [];

      const particleCount = 60;
      const connectionDistance = 120;
      const mouseDistance = 150;
      const particleColor = 'rgba(218, 165, 32, 0.6)';
      const lineColor = 'rgba(218, 165, 32, 0.2)';

      const mouse = { x: null, y: null };

      class Particle {
        constructor() {
          this.x = Math.random() * width;
          this.y = Math.random() * height;
          this.vx = (Math.random() - 0.5) * 0.5;
          this.vy = (Math.random() - 0.5) * 0.5;
          this.size = Math.random() * 2 + 1;
        }

        update() {
          this.x += this.vx;
          this.y += this.vy;

          if (this.x < 0 || this.x > width) this.vx *= -1;
          if (this.y < 0 || this.y > height) this.vy *= -1;

          const dx = mouse.x - this.x;
          const dy = mouse.y - this.y;
          const distance = Math.sqrt(dx * dx + dy * dy);

          if (distance < mouseDistance) {
            const forceDirectionX = dx / distance;
            const forceDirectionY = dy / distance;
            const force = (mouseDistance - distance) / mouseDistance;
            const directionX = forceDirectionX * force * this.size;
            const directionY = forceDirectionY * force * this.size;

            void directionX;
            void directionY;
          }
        }

        draw() {
          ctx.beginPath();
          ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
          ctx.fillStyle = particleColor;
          ctx.fill();
        }
      }

      function resize() {
        width = canvas.parentElement.offsetWidth;
        height = canvas.parentElement.offsetHeight;
        canvas.width = width;
        canvas.height = height;
      }

      function init() {
        resize();
        particles = [];
        for (let i = 0; i < particleCount; i++) {
          particles.push(new Particle());
        }
      }

      function animate() {
        requestAnimationFrame(animate);
        ctx.clearRect(0, 0, width, height);

        for (let i = 0; i < particles.length; i++) {
          particles[i].update();
          particles[i].draw();

          for (let j = i; j < particles.length; j++) {
            const dx = particles[i].x - particles[j].x;
            const dy = particles[i].y - particles[j].y;
            const distance = Math.sqrt(dx * dx + dy * dy);

            if (distance < connectionDistance) {
              ctx.beginPath();
              ctx.strokeStyle = lineColor;
              ctx.lineWidth = 1;
              ctx.moveTo(particles[i].x, particles[i].y);
              ctx.lineTo(particles[j].x, particles[j].y);
              ctx.stroke();
              ctx.closePath();
            }
          }
        }
      }

      window.addEventListener('resize', () => {
        resize();
        init();
      });

      canvas.addEventListener('mousemove', (e) => {
        const rect = canvas.getBoundingClientRect();
        mouse.x = e.clientX - rect.left;
        mouse.y = e.clientY - rect.top;
      });

      canvas.addEventListener('mouseleave', () => {
        mouse.x = undefined;
        mouse.y = undefined;
      });

      init();
      animate();
    } catch (error) {
      console.error('Animation script error:', error);
    }
  }

  if (document.readyState === 'complete' || document.readyState === 'interactive') {
    startAnimation();
  } else {
    window.addEventListener('load', startAnimation);
  }
})();
