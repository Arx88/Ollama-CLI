// Add event listener to add to cart buttons
document.querySelectorAll('.watch-card button').forEach((button) => {
  button.addEventListener('click', () => {
    // Get product name and price from watch card
    const productName = button.parentNode.querySelector('h3').textContent;
    const productPrice = parseFloat(
      button.parentNode.querySelector('p').textContent.replace('$', ''),
    );

    // Add product to shopping cart table
    const row = document.createElement('tr');
    row.innerHTML = `
            <td>${productName}</td>
            <td>$${productPrice.toFixed(2)}</td>
            <td>1</td>
            <td>$${productPrice.toFixed(2)}</td>
        `;
    document.querySelector('#shopping-cart table').appendChild(row);
  });
});
