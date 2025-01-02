import requests
import json
from datetime import datetime, timedelta

class SonarDashboard:
    def __init__(self, sonar_url, project_key):
        self.sonar_url = sonar_url
        self.project_key = project_key
        self.auth = ('admin', 'sqa_e80b25b6c0685599c12afcf17ee8767ad4463d38')

    def get_metrics(self):
        """Temel metrikleri al"""
        metrics = [
            'bugs',
            'vulnerabilities',
            'code_smells',
            'coverage',
            'duplicated_lines_density',
            'ncloc',
            'reliability_rating',
            'security_rating',
            'sqale_rating'
        ]
        
        endpoint = f"{self.sonar_url}/api/measures/component"
        params = {
            'component': self.project_key,
            'metricKeys': ','.join(metrics)
        }
        
        response = requests.get(endpoint, auth=self.auth, params=params)
        return response.json()

    def generate_html_report(self):
        """HTML formatında dashboard oluştur"""
        try:
            metrics = self.get_metrics()
            
            html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>SonarQube Dashboard</title>
                <style>
                    .dashboard {
                        display: grid;
                        grid-template-columns: repeat(3, 1fr);
                        gap: 20px;
                        padding: 20px;
                        background: #1a1a1a;
                        color: #ffffff;
                    }
                    .widget {
                        background: #2d2d2d;
                        padding: 15px;
                        border-radius: 8px;
                        box-shadow: 0 0 10px rgba(0,255,0,0.2);
                    }
                    .metric {
                        font-size: 24px;
                        color: #00ff00;
                    }
                    .rating {
                        font-size: 36px;
                        font-weight: bold;
                    }
                    .rating-A { color: #00ff00; }
                    .rating-B { color: #a4d93c; }
                    .rating-C { color: #d9a73c; }
                    .rating-D { color: #e2534f; }
                    .rating-E { color: #cc0000; }
                </style>
            </head>
            <body>
                <div class="dashboard">
            """
            
            # Temel metrikler
            if 'component' in metrics and 'measures' in metrics['component']:
                for measure in metrics['component']['measures']:
                    metric_name = measure['metric']
                    value = measure['value']
                    
                    html += f"""
                        <div class="widget">
                            <h3>{metric_name.replace('_', ' ').title()}</h3>
                            <div class="metric">
                                {self.format_metric(metric_name, value)}
                            </div>
                        </div>
                    """
            
            # Kod kapsama widget'ı
            html = self.add_code_coverage_widget(html)
            
            # Kod kalitesi özeti
            html = self.add_code_quality_summary(html)
            
            # Quality Gate durumu
            html = self.add_quality_gate(html)
            
            html += """
                </div>
            </body>
            </html>
            """
            
            return html
        except Exception as e:
            print(f"HTML rapor oluşturma hatası: {str(e)}")
            return """
            <div class="dashboard">
                <div class="widget">
                    <h3>SonarQube Bağlantı Hatası</h3>
                    <div class="metric">
                        SonarQube sunucusuna bağlanılamadı veya metrikler alınamadı.
                    </div>
                </div>
            </div>
            """

    def format_metric(self, metric_name, value):
        """Metrik değerlerini formatla"""
        if metric_name.endswith('_rating'):
            rating = 'ABCDE'[int(float(value)) - 1]
            return f'<span class="rating rating-{rating}">{rating}</span>'
        elif metric_name.endswith('_density'):
            return f'{float(value):.1f}%'
        else:
            return value 

    def add_trend_chart(self, html):
        """Metrik trendlerini gösteren grafik ekle"""
        endpoint = f"{self.sonar_url}/api/measures/search_history"
        params = {
            'component': self.project_key,
            'metrics': 'bugs,vulnerabilities,code_smells',
            'from': (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        }
        
        response = requests.get(endpoint, auth=self.auth, params=params)
        history = response.json()
        
        # Chart.js ile trend grafiği
        chart_data = self.prepare_chart_data(history)
        
        html += """
        <div class="widget full-width">
            <h3>Trend Analysis</h3>
            <canvas id="trendChart"></canvas>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <script>
                const ctx = document.getElementById('trendChart').getContext('2d');
                new Chart(ctx, %s);
            </script>
        </div>
        """ % json.dumps(chart_data)
        
        return html 

    def add_quality_gate(self, html):
        """Quality Gate durumunu ekle"""
        endpoint = f"{self.sonar_url}/api/qualitygates/project_status"
        params = {'projectKey': self.project_key}
        
        response = requests.get(endpoint, auth=self.auth, params=params)
        status = response.json()['projectStatus']['status']
        
        html += f"""
        <div class="widget quality-gate">
            <h3>Quality Gate</h3>
            <div class="status status-{status.lower()}">
                {status}
            </div>
        </div>
        """
        
        return html 

    def add_code_coverage_widget(self, html):
        """Kod kapsama oranı widget'ı"""
        coverage_data = self.get_coverage_metrics()
        
        html += """
        <div class="widget coverage-widget">
            <h3>Kod Kapsama</h3>
            <div class="coverage-grid">
                <div class="coverage-item">
                    <div class="coverage-label">Genel Kapsama</div>
                    <div class="coverage-value">{:.1f}%</div>
                    <div class="coverage-bar">
                        <div class="bar" style="width: {}%"></div>
                    </div>
                </div>
                <div class="coverage-details">
                    <div>Test Edilen Satırlar: {}</div>
                    <div>Toplam Satır: {}</div>
                </div>
            </div>
        </div>
        """.format(
            coverage_data['coverage'],
            coverage_data['coverage'],
            coverage_data['covered_lines'],
            coverage_data['total_lines']
        )
        return html

    def add_code_quality_summary(self, html):
        """Kod kalitesi özeti widget'ı"""
        quality_data = self.get_quality_metrics()
        
        html += """
        <div class="widget quality-summary">
            <h3>Kod Kalitesi Özeti</h3>
            <div class="quality-grid">
                <div class="quality-item bugs">
                    <i class="fas fa-bug"></i>
                    <span class="count">{}</span>
                    <span class="label">Hatalar</span>
                </div>
                <div class="quality-item vulnerabilities">
                    <i class="fas fa-shield-alt"></i>
                    <span class="count">{}</span>
                    <span class="label">Güvenlik Açıkları</span>
                </div>
                <div class="quality-item code-smells">
                    <i class="fas fa-code"></i>
                    <span class="count">{}</span>
                    <span class="label">Kod Kokuları</span>
                </div>
            </div>
        </div>
        """.format(
            quality_data['bugs'],
            quality_data['vulnerabilities'],
            quality_data['code_smells']
        )
        return html 

    def get_coverage_metrics(self):
        """Test coverage metriklerini al"""
        try:
            endpoint = f"{self.sonar_url}/api/measures/component"
            params = {
                'component': self.project_key,
                'metricKeys': 'coverage,lines_to_cover,uncovered_lines'
            }
            
            response = requests.get(endpoint, auth=self.auth, params=params)
            data = response.json()
            
            coverage = 0.0
            covered_lines = 0
            total_lines = 0
            
            for measure in data['component']['measures']:
                if measure['metric'] == 'coverage':
                    coverage = float(measure['value'])
                elif measure['metric'] == 'lines_to_cover':
                    total_lines = int(measure['value'])
                elif measure['metric'] == 'uncovered_lines':
                    covered_lines = total_lines - int(measure['value'])
            
            return {
                'coverage': coverage,
                'covered_lines': covered_lines,
                'total_lines': total_lines
            }
        except Exception as e:
            print(f"Coverage metrikleri alınamadı: {str(e)}")
            return {
                'coverage': 0.0,
                'covered_lines': 0,
                'total_lines': 0
            }

    def get_quality_metrics(self):
        """Kod kalitesi metriklerini al"""
        try:
            endpoint = f"{self.sonar_url}/api/measures/component"
            params = {
                'component': self.project_key,
                'metricKeys': 'bugs,vulnerabilities,code_smells'
            }
            
            response = requests.get(endpoint, auth=self.auth, params=params)
            data = response.json()
            
            metrics = {
                'bugs': 0,
                'vulnerabilities': 0,
                'code_smells': 0
            }
            
            for measure in data['component']['measures']:
                if measure['metric'] in metrics:
                    metrics[measure['metric']] = int(measure['value'])
            
            return metrics
        except Exception as e:
            print(f"Kalite metrikleri alınamadı: {str(e)}")
            return {
                'bugs': 0,
                'vulnerabilities': 0,
                'code_smells': 0
            } 