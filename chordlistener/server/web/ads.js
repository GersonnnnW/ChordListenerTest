"use strict";

// Advertising hook. This file is only loaded when the server runs with
// CL_ADS=true; until then the page reserves no space for ads at all.
//
// The page has three ad spaces, each an <aside class="ad-slot"> element:
//
//   data-ad-slot="top"     under the header, on every screen
//   data-ad-slot="result"  below a finished transcription
//   data-ad-slot="bottom"  above the footer
//
// Each reserves room for a 728x90 banner (320x50 on phones), so the page does
// not jump when an ad loads. Empty spaces show a dashed placeholder so you can
// see the layout before any ad network is connected.
//
// To connect a network, put its code below and fill the spaces, for example:
//
//   for (const slot of document.querySelectorAll("[data-ad-slot]")) {
//     slot.innerHTML = '<ins class="adsbygoogle" data-ad-client="ca-pub-XXXX" data-ad-slot="YYYY" style="display:block"></ins>';
//   }
//
// and list the network's domains in CL_AD_ORIGINS (space separated), e.g.
//   CL_AD_ORIGINS="https://pagead2.googlesyndication.com https://*.doubleclick.net"
// so the page's content security policy lets its scripts, frames and images load.
// Ads are never shown in embed mode (?embed=1) or in printed/downloaded sheet music.
